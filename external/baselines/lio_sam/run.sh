#!/bin/bash
# Run LIO-SAM on a KITTI Odometry sequence.
# Usage: run.sh <sequence_id>
#
# Expects a pre-built rosbag at /data/bags/kitti_${SEQ}_fixed.bag, produced
# by slam-baselines/bag-builder (see prepare_bags.sh). The bag is read-only
# and shared across all three baselines for evaluation consistency.
#
# SUP-01 Stage C: LiDAR-IMU extrinsic is read from /data/kitti_raw at
# container start and injected into /config/params.yaml via envsubst,
# replacing the build-time inject_kitti_extrinsic.py pipeline. One image
# (slam-baselines/lio_sam:latest) now serves every sequence.
set -e

SEQ="${1:?Usage: run.sh <sequence_id>}"
OUTPUT_DIR="/output"
# BAG_VARIANT selects which cache bag to consume (see prepare_bag.sh).
# Default "_navaccel" = kitti_${SEQ}_fixed_navaccel.bag (SUP-01 α fallback).
# Set BAG_VARIANT="" to use body-frame accel bag kitti_${SEQ}_fixed.bag.
BAG_VARIANT="${BAG_VARIANT:-_navaccel}"
BAG="/data/bags/kitti_${SEQ}_fixed${BAG_VARIANT}.bag"
ODOM_BAG="/tmp/odom_${SEQ}.bag"

echo "=========================================="
echo "LIO-SAM — Sequence ${SEQ} (BAG_VARIANT=${BAG_VARIANT})"
echo "  input bag: ${BAG}"
echo "=========================================="

if [ ! -f "${BAG}" ]; then
    echo "FAIL: cached bag not found at ${BAG}" >&2
    echo "      run prepare_bags.sh first to build it" >&2
    exit 1
fi

# Render extrinsic into params.yaml from KITTI Raw calib_imu_to_velo.txt.
# The calib is mounted read-only at /data/kitti_raw; per-sequence date
# mapping lives in inject_kitti_extrinsic.py (ODOM_TO_RAW_DATE).
if [ ! -d "/data/kitti_raw" ]; then
    echo "FAIL: /data/kitti_raw not mounted (needed for LiDAR-IMU extrinsic)" >&2
    echo "      mount with: -v \$KITTI_RAW_DIR:/data/kitti_raw:ro" >&2
    exit 1
fi

echo "[0/4] Rendering params.yaml from KITTI Raw calib (seq ${SEQ})..."
KITTI_EXT_ROT=$(python3 /scripts/extrinsic_to_env.py \
    --sequence "${SEQ}" --kitti-raw-dir /data/kitti_raw --field rot)
KITTI_EXT_TRANS=$(python3 /scripts/extrinsic_to_env.py \
    --sequence "${SEQ}" --kitti-raw-dir /data/kitti_raw --field trans)
export KITTI_EXT_ROT KITTI_EXT_TRANS
echo "  KITTI_EXT_TRANS = [${KITTI_EXT_TRANS}]"
echo "  KITTI_EXT_ROT   = [${KITTI_EXT_ROT}]"
envsubst < /config/params.yaml.tpl > /config/params.yaml

source /catkin_ws/devel/setup.bash
export ROS_MASTER_URI=http://localhost:11311

# Start roscore
roscore &
ROSCORE_PID=$!
sleep 2

# Launch LIO-SAM
echo "[1/4] Launching LIO-SAM..."
roslaunch /launch/kitti.launch &
SLAM_PID=$!
sleep 8

echo "[2/4] Recording odometry and playing cached bag..."
rosbag record -O "${ODOM_BAG}" /lio_sam/mapping/odometry &
RECORD_PID=$!
sleep 1

rosbag play "${BAG}" --clock -r 0.5
echo "  Bag playback finished. Waiting..."
sleep 15

kill ${RECORD_PID} 2>/dev/null || true
sleep 2

# Extract poses
echo "[3/4] Extracting poses..."
python3 /scripts/extract_poses.py \
    --odom-bag "${ODOM_BAG}" \
    --odom-topic /lio_sam/mapping/odometry \
    --lidar-bag "${BAG}" \
    --output "${OUTPUT_DIR}/poses_${SEQ}.txt"

# Cleanup (leave cached bag in place)
rm -f "${ODOM_BAG}"
kill ${SLAM_PID} 2>/dev/null || true
kill ${ROSCORE_PID} 2>/dev/null || true

echo "[4/4] Done."
if [ -f "${OUTPUT_DIR}/poses_${SEQ}.txt" ]; then
    LINES=$(wc -l < "${OUTPUT_DIR}/poses_${SEQ}.txt")
    echo "SUCCESS: ${LINES} poses written"
else
    echo "FAIL: No output poses"
    exit 1
fi
