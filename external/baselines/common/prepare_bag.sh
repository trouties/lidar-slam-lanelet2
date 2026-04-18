#!/bin/bash
# Entry point for slam-baselines/bag-builder. Runs inside the container.
#
# Steps (for a given KITTI Odometry sequence):
#   1. Convert KITTI Odometry + KITTI Raw OxTS IMU to a rosbag with ring fields
#      (kitti_to_rosbag.py --with-imu --add-ring). Ring fields are required by
#      LIO-SAM's imageProjection; FAST-LIO2 and hdl_graph_slam ignore them.
#   2. Forward-propagate IMU header.stamp to strict monotonicity
#      (fix_imu_timestamps.py). GTSAM preintegration in LIO-SAM throws on
#      dt <= 0 and KITTI OxTS timestamps.txt contains occasional duplicates.
#   3. Emit the repaired bag to /cache/kitti_${SEQ}_fixed.bag.
#
# The caller (host-side prepare_bags.sh) mounts:
#   /data/kitti       = KITTI Odometry root   (read-only)
#   /data/kitti_raw   = KITTI Raw root        (read-only)
#   /cache            = persistent bag cache  (writable)
set -e

SEQ="${1:?Usage: prepare_bag.sh <sequence>}"
# RING_MODE=yes (default): include ring + per-point time fields (LIO-SAM needs).
# RING_MODE=no: plain xyz+intensity. Used by R2 bag bisect to test whether
# per-point time is confusing FAST-LIO2 / hdl_graph_slam.
#
# ACCEL_MODE=body (default): cols 14-16 OxTS body-frame specific force.
# ACCEL_MODE=nav:  cols 11-13 OxTS nav-frame kinematic accel (gravity
#                  pre-subtracted). SUP-01 α fallback — physically wrong
#                  for body-frame GTSAM integration but empirically the
#                  content that gives LIO-SAM the historical 27m result.
#
# Output file naming reflects the selected modes:
#   default (ring=yes, accel=body) → kitti_${SEQ}_fixed.bag
#   ring=no                        → kitti_${SEQ}_fixed_noring.bag
#   accel=nav                      → kitti_${SEQ}_fixed_navaccel.bag
#   ring=no + accel=nav            → kitti_${SEQ}_fixed_noring_navaccel.bag
RING_MODE="${RING_MODE:-yes}"
ACCEL_MODE="${ACCEL_MODE:-body}"
# ZERO_TIME=yes: per-point time field forced to 0. KITTI is pre-deskewed,
# so azimuth-based times cause LIO-SAM double-compensation and ~20x APE
# regression. The stage0-v3 reference (SE(3) APE 27m) used zero times.
ZERO_TIME="${ZERO_TIME:-no}"
KITTI_DIR="/data/kitti"
KITTI_RAW_DIR="/data/kitti_raw"
CACHE_DIR="/cache"
TMP_BAG="/tmp/kitti_${SEQ}_raw.bag"

SUFFIX=""
RING_ARG="--add-ring"
ZERO_TIME_ARG=""
if [ "${RING_MODE}" = "no" ]; then
    SUFFIX="${SUFFIX}_noring"
    RING_ARG=""
fi
if [ "${ACCEL_MODE}" = "nav" ]; then
    SUFFIX="${SUFFIX}_navaccel"
elif [ "${ACCEL_MODE}" != "body" ]; then
    echo "FAIL: ACCEL_MODE must be 'body' or 'nav', got '${ACCEL_MODE}'" >&2
    exit 1
fi
if [ "${ZERO_TIME}" = "yes" ]; then
    SUFFIX="${SUFFIX}_zerotime"
    ZERO_TIME_ARG="--zero-time"
elif [ "${ZERO_TIME}" != "no" ]; then
    echo "FAIL: ZERO_TIME must be 'yes' or 'no', got '${ZERO_TIME}'" >&2
    exit 1
fi
OUT_BAG="${CACHE_DIR}/kitti_${SEQ}_fixed${SUFFIX}.bag"

echo "=========================================="
echo "bag-builder — Sequence ${SEQ} (RING=${RING_MODE}, ACCEL=${ACCEL_MODE}, ZERO_TIME=${ZERO_TIME})"
echo "  output: ${OUT_BAG}"
echo "=========================================="

mkdir -p "${CACHE_DIR}"

echo "[1/2] Converting KITTI to rosbag (imu, ring=${RING_MODE}, accel=${ACCEL_MODE}, zero_time=${ZERO_TIME})..."
python3 /scripts/kitti_to_rosbag.py \
    --kitti-dir "${KITTI_DIR}" --sequence "${SEQ}" \
    --output "${TMP_BAG}" --with-imu --kitti-raw-dir "${KITTI_RAW_DIR}" \
    --accel-mode "${ACCEL_MODE}" ${RING_ARG} ${ZERO_TIME_ARG}

echo "[2/2] Repairing IMU timestamps (forward-propagate monotonicity)..."
python3 /scripts/fix_imu_timestamps.py \
    --input "${TMP_BAG}" --output "${OUT_BAG}"

rm -f "${TMP_BAG}"

if [ -f "${OUT_BAG}" ]; then
    SIZE=$(stat -c %s "${OUT_BAG}")
    echo "SUCCESS: cached bag ${OUT_BAG} (${SIZE} bytes)"
else
    echo "FAIL: cached bag not produced" >&2
    exit 1
fi
