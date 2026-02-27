#!/bin/bash
# Run all external SLAM baselines on KITTI Odometry sequences.
#
# Usage:
#   ./run_all_baselines.sh [sequences] [systems]
#   ./run_all_baselines.sh "00,05" "hdl_graph_slam,fast_lio2,lio_sam"
#
# Prerequisites:
#   - Docker images built (make build-all)
#   - KITTI Odometry data at ~/data/kitti/odometry/dataset/
#   - KITTI Raw OxTS at ~/data/kitti_raw/ (for LIO-SAM, FAST-LIO2)
set -e

SEQUENCES="${1:-00,05}"
SYSTEMS="${2:-hdl_graph_slam,fast_lio2,lio_sam}"
KITTI_DIR="${KITTI_DIR:-$HOME/data/kitti/odometry/dataset}"
KITTI_RAW_DIR="${KITTI_RAW_DIR:-$HOME/data/kitti_raw}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUTPUT_BASE="${SCRIPT_DIR}"

echo "============================================================"
echo "SUP-01 External Baseline Runner"
echo "  Sequences: ${SEQUENCES}"
echo "  Systems:   ${SYSTEMS}"
echo "  KITTI:     ${KITTI_DIR}"
echo "  Raw IMU:   ${KITTI_RAW_DIR}"
echo "============================================================"

TOTAL=0
SUCCESS=0
FAIL=0

for system in ${SYSTEMS//,/ }; do
    for seq in ${SEQUENCES//,/ }; do
        TOTAL=$((TOTAL + 1))
        echo ""
        echo ">>> Running ${system} on seq ${seq} ..."

        OUTPUT_DIR="${OUTPUT_BASE}/${system}/results"
        mkdir -p "${OUTPUT_DIR}"

        if docker run --rm \
            -v "${KITTI_DIR}:/data/kitti:ro" \
            -v "${KITTI_RAW_DIR}:/data/kitti_raw:ro" \
            -v "${OUTPUT_DIR}:/output" \
            --memory=12g \
            --cpus=14 \
            "slam-baselines/${system}:latest" \
            "${seq}" 2>&1 | tee "/tmp/baseline_${system}_${seq}.log"; then

            if [ -f "${OUTPUT_DIR}/poses_${seq}.txt" ]; then
                LINES=$(wc -l < "${OUTPUT_DIR}/poses_${seq}.txt")
                echo ">>> OK: ${system} seq ${seq} — ${LINES} poses"
                SUCCESS=$((SUCCESS + 1))
            else
                echo ">>> FAIL: ${system} seq ${seq} — no output file"
                FAIL=$((FAIL + 1))
            fi
        else
            echo ">>> FAIL: ${system} seq ${seq} — container exited with error"
            FAIL=$((FAIL + 1))
        fi
    done
done

echo ""
echo "============================================================"
echo "Summary: ${SUCCESS}/${TOTAL} succeeded, ${FAIL} failed"
echo "============================================================"

if [ ${FAIL} -gt 0 ]; then
    echo "Check logs in /tmp/baseline_*.log"
    exit 1
fi
