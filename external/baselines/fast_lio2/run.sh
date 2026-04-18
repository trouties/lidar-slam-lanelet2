#!/bin/bash
# Run FAST-LIO2 on a KITTI Odometry sequence.
# Usage: run.sh <sequence_id>
#
# Expects a pre-built rosbag at /data/bags/kitti_${SEQ}_fixed.bag, produced
# by slam-baselines/bag-builder (see prepare_bags.sh). The bag is read-only
# and shared across all three baselines for evaluation consistency.
set -e

SEQ="${1:?Usage: run.sh <sequence_id>}"
OUTPUT_DIR="/output"
BAG="/data/bags/kitti_${SEQ}_fixed.bag"
ODOM_BAG="/tmp/odom_${SEQ}.bag"

echo "=========================================="
echo "FAST-LIO2 — Sequence ${SEQ}"
echo "=========================================="

if [ ! -f "${BAG}" ]; then
    echo "FAIL: cached bag not found at ${BAG}" >&2
    echo "      run prepare_bags.sh first to build it" >&2
    exit 1
fi

source /catkin_ws/devel/setup.bash
export ROS_MASTER_URI=http://localhost:11311

# Start roscore
roscore &
ROSCORE_PID=$!
sleep 2

# Launch FAST-LIO2
echo "[1/3] Launching FAST-LIO2..."
roslaunch /launch/kitti.launch &
SLAM_PID=$!
sleep 8  # give FAST-LIO2 time to subscribe

echo "[2/3] Recording odometry and playing cached bag..."
rosbag record -O "${ODOM_BAG}" /Odometry &
RECORD_PID=$!
sleep 1

rosbag play "${BAG}" --clock -r 0.5
echo "  Bag playback finished. Waiting..."
sleep 10

kill ${RECORD_PID} 2>/dev/null || true
sleep 2

# Extract poses
echo "[3/3] Extracting poses..."
python3 /scripts/extract_poses.py \
    --odom-bag "${ODOM_BAG}" \
    --odom-topic /Odometry \
    --lidar-bag "${BAG}" \
    --output "${OUTPUT_DIR}/poses_${SEQ}.txt"

# Cleanup (leave cached bag in place)
rm -f "${ODOM_BAG}"
kill ${SLAM_PID} 2>/dev/null || true
kill ${ROSCORE_PID} 2>/dev/null || true

if [ -f "${OUTPUT_DIR}/poses_${SEQ}.txt" ]; then
    LINES=$(wc -l < "${OUTPUT_DIR}/poses_${SEQ}.txt")
    echo "SUCCESS: ${LINES} poses written"
else
    echo "FAIL: No output poses"
    exit 1
fi
