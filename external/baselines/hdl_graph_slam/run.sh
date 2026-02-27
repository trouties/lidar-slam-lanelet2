#!/bin/bash
# Run hdl_graph_slam on a KITTI Odometry sequence.
# Usage: run.sh <sequence_id>
# Expects: /data/kitti mounted (KITTI Odometry root), /output mounted (results dir)
set -e

SEQ="${1:?Usage: run.sh <sequence_id>}"
KITTI_DIR="/data/kitti"
OUTPUT_DIR="/output"
BAG="/tmp/kitti_${SEQ}.bag"
ODOM_BAG="/tmp/odom_${SEQ}.bag"

echo "=========================================="
echo "hdl_graph_slam — Sequence ${SEQ}"
echo "=========================================="

source /catkin_ws/devel/setup.bash
export ROS_MASTER_URI=http://localhost:11311

# Step 1: Start roscore
roscore &
ROSCORE_PID=$!
sleep 2

# Step 2: Convert KITTI to rosbag (LiDAR only)
echo "[1/4] Converting KITTI to rosbag..."
python3 /scripts/kitti_to_rosbag.py \
    --kitti-dir "${KITTI_DIR}" --sequence "${SEQ}" \
    --output "${BAG}" --lidar-only

# Step 3: Launch hdl_graph_slam + record odometry
echo "[2/4] Launching hdl_graph_slam..."
roslaunch /launch/kitti.launch &
SLAM_PID=$!
sleep 5

echo "[3/4] Recording odometry and playing bag..."
rosbag record -O "${ODOM_BAG}" /odom &
RECORD_PID=$!
sleep 1

# Play at half speed to give processing headroom
rosbag play "${BAG}" --clock -r 0.5 --wait-for-subscribers
echo "  Bag playback finished. Waiting for final processing..."
sleep 15  # allow graph optimization to finish

# Stop recording
kill ${RECORD_PID} 2>/dev/null || true
sleep 2

# Step 4: Extract poses
echo "[4/4] Extracting poses..."
python3 /scripts/extract_poses.py \
    --odom-bag "${ODOM_BAG}" \
    --odom-topic /odom \
    --lidar-bag "${BAG}" \
    --output "${OUTPUT_DIR}/poses_${SEQ}.txt"

# Cleanup
rm -f "${BAG}" "${ODOM_BAG}"
kill ${SLAM_PID} 2>/dev/null || true
kill ${ROSCORE_PID} 2>/dev/null || true

# Verify
if [ -f "${OUTPUT_DIR}/poses_${SEQ}.txt" ]; then
    LINES=$(wc -l < "${OUTPUT_DIR}/poses_${SEQ}.txt")
    echo "SUCCESS: ${LINES} poses written to ${OUTPUT_DIR}/poses_${SEQ}.txt"
else
    echo "FAIL: No output poses"
    exit 1
fi
