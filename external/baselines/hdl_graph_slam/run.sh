#!/bin/bash
# Run hdl_graph_slam on a KITTI Odometry sequence.
# Usage: run.sh <sequence_id>
#
# Expects a pre-built rosbag at /data/bags/kitti_${SEQ}_fixed.bag, produced
# by slam-baselines/bag-builder (see prepare_bags.sh). The bag carries both
# /velodyne_points (with ring fields) and /imu/data, but hdl_graph_slam only
# subscribes to /velodyne_points, so the extra IMU topic is harmless.
set -e

SEQ="${1:?Usage: run.sh <sequence_id>}"
OUTPUT_DIR="/output"
BAG="/data/bags/kitti_${SEQ}_fixed.bag"
ODOM_BAG="/tmp/odom_${SEQ}.bag"

echo "=========================================="
echo "hdl_graph_slam — Sequence ${SEQ}"
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

# Launch hdl_graph_slam
echo "[1/3] Launching hdl_graph_slam..."
roslaunch /launch/kitti.launch &
SLAM_PID=$!
sleep 5

echo "[2/3] Recording odometry and playing cached bag..."
rosbag record -O "${ODOM_BAG}" /odom &
RECORD_PID=$!
sleep 1

# --wait-for-subscribers deadlocks with use_sim_time=true because the
# subscriber nodelet won't spin up until /clock is published, and /clock
# isn't published until rosbag play starts (SUP-01 P0-1 Stage D). Drop the
# flag; the 5 s sleep above gives nodelets enough time to subscribe.
rosbag play "${BAG}" --clock -r 0.5
echo "  Bag playback finished. Waiting for final processing..."
sleep 15

kill ${RECORD_PID} 2>/dev/null || true
sleep 2

# Extract poses
echo "[3/3] Extracting poses..."
python3 /scripts/extract_poses.py \
    --odom-bag "${ODOM_BAG}" \
    --odom-topic /odom \
    --lidar-bag "${BAG}" \
    --output "${OUTPUT_DIR}/poses_${SEQ}.txt"

# Cleanup (leave cached bag in place)
rm -f "${ODOM_BAG}"
kill ${SLAM_PID} 2>/dev/null || true
kill ${ROSCORE_PID} 2>/dev/null || true

if [ -f "${OUTPUT_DIR}/poses_${SEQ}.txt" ]; then
    LINES=$(wc -l < "${OUTPUT_DIR}/poses_${SEQ}.txt")
    echo "SUCCESS: ${LINES} poses written to ${OUTPUT_DIR}/poses_${SEQ}.txt"
else
    echo "FAIL: No output poses"
    exit 1
fi
