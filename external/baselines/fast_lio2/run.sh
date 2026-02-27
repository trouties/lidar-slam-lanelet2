#!/bin/bash
# Run FAST-LIO2 on a KITTI Odometry sequence.
# Usage: run.sh <sequence_id>
set -e

SEQ="${1:?Usage: run.sh <sequence_id>}"
KITTI_DIR="/data/kitti"
KITTI_RAW_DIR="/data/kitti_raw"
OUTPUT_DIR="/output"
BAG="/tmp/kitti_${SEQ}.bag"
ODOM_BAG="/tmp/odom_${SEQ}.bag"

echo "=========================================="
echo "FAST-LIO2 — Sequence ${SEQ}"
echo "=========================================="

source /catkin_ws/devel/setup.bash
export ROS_MASTER_URI=http://localhost:11311

# Step 1: Start roscore
roscore &
ROSCORE_PID=$!
sleep 2

# Step 2: Convert KITTI to rosbag (with IMU)
echo "[1/4] Converting KITTI to rosbag (with IMU)..."
python3 /scripts/kitti_to_rosbag.py \
    --kitti-dir "${KITTI_DIR}" --sequence "${SEQ}" \
    --output "${BAG}" --with-imu --kitti-raw-dir "${KITTI_RAW_DIR}"

# Step 3: Launch FAST-LIO2 + record odometry
echo "[2/4] Launching FAST-LIO2..."
roslaunch /launch/kitti.launch &
SLAM_PID=$!
sleep 8  # give FAST-LIO2 time to subscribe

echo "[3/4] Recording odometry and playing bag..."
rosbag record -O "${ODOM_BAG}" /Odometry &
RECORD_PID=$!
sleep 1

# Don't use --wait-for-subscribers; sleep above gives enough time
rosbag play "${BAG}" --clock -r 0.5
echo "  Bag playback finished. Waiting..."
sleep 10

kill ${RECORD_PID} 2>/dev/null || true
sleep 2

# Step 4: Extract poses
echo "[4/4] Extracting poses..."
python3 /scripts/extract_poses.py \
    --odom-bag "${ODOM_BAG}" \
    --odom-topic /Odometry \
    --lidar-bag "${BAG}" \
    --output "${OUTPUT_DIR}/poses_${SEQ}.txt"

# Cleanup
rm -f "${BAG}" "${ODOM_BAG}"
kill ${SLAM_PID} 2>/dev/null || true
kill ${ROSCORE_PID} 2>/dev/null || true

if [ -f "${OUTPUT_DIR}/poses_${SEQ}.txt" ]; then
    LINES=$(wc -l < "${OUTPUT_DIR}/poses_${SEQ}.txt")
    echo "SUCCESS: ${LINES} poses written"
else
    echo "FAIL: No output poses"
    exit 1
fi
