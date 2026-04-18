#!/bin/bash
# SUP-01 P0-1: host-side wrapper that populates the shared KITTI bag cache
# used by all three baselines. Converts KITTI Odometry + Raw OxTS to a
# ring-enabled ROS bag and repairs non-monotonic IMU timestamps.
#
# Cache layout: $KITTI_BAG_CACHE/kitti_${SEQ}_fixed.bag (default cache dir
# is ~/data/kitti_bags_cache). Rebuilds only bags that are missing unless
# FORCE_REBUILD=1 is set. The LIO-SAM / FAST-LIO2 / hdl_graph_slam run
# scripts mount this directory read-only at /data/bags inside their
# containers, so they all consume identical bits.
#
# Usage:
#   ./prepare_bags.sh "00"            # build seq 00 if missing
#   ./prepare_bags.sh "00,05"         # build seq 00 and 05 if missing
#   FORCE_REBUILD=1 ./prepare_bags.sh "00"
set -e

SEQUENCES="${1:-00,05}"
KITTI_DIR="${KITTI_DIR:-$HOME/data/kitti/odometry/dataset}"
KITTI_RAW_DIR="${KITTI_RAW_DIR:-$HOME/data/kitti_raw}"
KITTI_BAG_CACHE="${KITTI_BAG_CACHE:-$HOME/data/kitti_bags_cache}"
FORCE_REBUILD="${FORCE_REBUILD:-0}"

mkdir -p "${KITTI_BAG_CACHE}"

echo "============================================================"
echo "SUP-01 bag cache preparation"
echo "  Sequences: ${SEQUENCES}"
echo "  KITTI:     ${KITTI_DIR}"
echo "  Raw IMU:   ${KITTI_RAW_DIR}"
echo "  Cache:     ${KITTI_BAG_CACHE}"
echo "  Force:     ${FORCE_REBUILD}"
echo "============================================================"

# Build both bag variants the baselines consume:
#   kitti_${seq}_fixed.bag           — ACCEL_MODE=body  (FAST-LIO2, hdl_graph_slam)
#   kitti_${seq}_fixed_navaccel.bag  — ACCEL_MODE=nav   (LIO-SAM α fallback, default BAG_VARIANT)
# Each variant is an independent bag-builder run with its own cache entry so
# the cache-hit shortcut applies per-variant.
build_variant() {
    local seq="$1" accel_mode="$2" suffix="$3"
    local bag_path="${KITTI_BAG_CACHE}/kitti_${seq}_fixed${suffix}.bag"

    if [ -f "${bag_path}" ] && [ "${FORCE_REBUILD}" != "1" ]; then
        local size
        size=$(stat -c %s "${bag_path}")
        echo ">>> seq ${seq} (${accel_mode}-accel): cache hit (${size} bytes) — ${bag_path}"
        return 0
    fi

    if [ "${FORCE_REBUILD}" = "1" ] && [ -f "${bag_path}" ]; then
        echo ">>> seq ${seq} (${accel_mode}-accel): FORCE_REBUILD — removing old cache"
        docker run --rm \
            -v "${KITTI_BAG_CACHE}:/cache" \
            --entrypoint /bin/sh \
            slam-baselines/bag-builder:latest \
            -c "rm -f ${bag_path//${KITTI_BAG_CACHE}/\/cache}"
    fi

    echo ">>> seq ${seq} (${accel_mode}-accel): building cache bag via slam-baselines/bag-builder..."
    docker run --rm \
        -e "ACCEL_MODE=${accel_mode}" \
        -v "${KITTI_DIR}:/data/kitti:ro" \
        -v "${KITTI_RAW_DIR}:/data/kitti_raw:ro" \
        -v "${KITTI_BAG_CACHE}:/cache" \
        slam-baselines/bag-builder:latest \
        "${seq}"

    if [ ! -f "${bag_path}" ]; then
        echo ">>> FAIL: seq ${seq} (${accel_mode}-accel) bag not produced at ${bag_path}" >&2
        exit 1
    fi
}

for seq in ${SEQUENCES//,/ }; do
    build_variant "${seq}" "body" ""
    build_variant "${seq}" "nav" "_navaccel"
done

echo "============================================================"
echo "bag cache ready"
ls -la "${KITTI_BAG_CACHE}"
echo "============================================================"
