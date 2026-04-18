#!/bin/bash
# Run all external SLAM baselines on KITTI Odometry sequences.
#
# Usage:
#   ./run_all_baselines.sh [sequences] [systems]
#   ./run_all_baselines.sh "00,05" "hdl_graph_slam,fast_lio2,lio_sam"
#
# Prerequisites:
#   - Docker images built (make build-all, includes bag-builder)
#   - KITTI Odometry data at ~/data/kitti/odometry/dataset/
#   - KITTI Raw OxTS at ~/data/kitti_raw/ (for LIO-SAM, FAST-LIO2)
#
# Shared bag cache: this script first calls prepare_bags.sh to populate
# $KITTI_BAG_CACHE with a single ring-enabled, IMU-timestamp-repaired bag
# per sequence, then mounts the cache dir read-only into each baseline
# container so all three systems evaluate on identical bits.
set -e

SEQUENCES="${1:-00,05}"
SYSTEMS="${2:-hdl_graph_slam,fast_lio2,lio_sam}"
KITTI_DIR="${KITTI_DIR:-$HOME/data/kitti/odometry/dataset}"
KITTI_RAW_DIR="${KITTI_RAW_DIR:-$HOME/data/kitti_raw}"
KITTI_BAG_CACHE="${KITTI_BAG_CACHE:-$HOME/data/kitti_bags_cache}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUTPUT_BASE="${SCRIPT_DIR}"

echo "============================================================"
echo "SUP-01 External Baseline Runner"
echo "  Sequences: ${SEQUENCES}"
echo "  Systems:   ${SYSTEMS}"
echo "  KITTI:     ${KITTI_DIR}"
echo "  Raw IMU:   ${KITTI_RAW_DIR}"
echo "  Bag cache: ${KITTI_BAG_CACHE}"
echo "============================================================"

# Ensure the shared bag cache is populated (idempotent; skips cache hits)
export KITTI_DIR KITTI_RAW_DIR KITTI_BAG_CACHE
bash "${SCRIPT_DIR}/prepare_bags.sh" "${SEQUENCES}"

# Parallel execution: for each sequence, launch all requested systems at the
# same time, each capped at DOCKER_CPUS_PER_SYSTEM cores. LIO-SAM / FAST-LIO2
# / hdl_graph_slam each use 2-3 cores in steady state, so 3 systems × 3 cores
# = 9 cores fits a 10-core host. Tune via DOCKER_CPUS_PER_SYSTEM.
TOTAL=0
SUCCESS=0
FAIL=0

SYSTEMS_ARR=(${SYSTEMS//,/ })
N_SYSTEMS=${#SYSTEMS_ARR[@]}
DOCKER_MEM="${DOCKER_MEM:-6g}"
DOCKER_CPUS_PER_SYSTEM="${DOCKER_CPUS_PER_SYSTEM:-3}"

for seq in ${SEQUENCES//,/ }; do
    echo ""
    echo ">>> launching ${N_SYSTEMS} systems in parallel for seq ${seq}"
    echo "    per-container cpus=${DOCKER_CPUS_PER_SYSTEM} mem=${DOCKER_MEM}"
    PIDS=()
    RC_FILES=()
    for system in "${SYSTEMS_ARR[@]}"; do
        TOTAL=$((TOTAL + 1))
        OUTPUT_DIR="${OUTPUT_BASE}/${system}/results"
        mkdir -p "${OUTPUT_DIR}"
        LOG="/tmp/baseline_${system}_${seq}.log"
        RC_FILE="/tmp/baseline_${system}_${seq}.rc"
        NAME="sup01-${system}-${seq}-$$"
        rm -f "${RC_FILE}"

        # Post Stage C all three baselines use a single :latest tag; LIO-SAM
        # derives its per-seq LiDAR-IMU extrinsic from KITTI Raw at runtime
        # (see external/baselines/lio_sam/run.sh + params.yaml.tpl). This
        # removes the docker-layer-cache ambiguity of per-seq image tags.
        IMAGE="slam-baselines/${system}:latest"

        (
            docker run --rm \
                --name "${NAME}" \
                -v "${KITTI_DIR}:/data/kitti:ro" \
                -v "${KITTI_RAW_DIR}:/data/kitti_raw:ro" \
                -v "${KITTI_BAG_CACHE}:/data/bags:ro" \
                -v "${OUTPUT_DIR}:/output" \
                --memory="${DOCKER_MEM}" \
                --cpus="${DOCKER_CPUS_PER_SYSTEM}" \
                "${IMAGE}" \
                "${seq}" > "${LOG}" 2>&1
            echo $? > "${RC_FILE}"
        ) &
        PIDS+=($!)
        RC_FILES+=("${system}|${seq}|${RC_FILE}|${OUTPUT_DIR}/poses_${seq}.txt|${LOG}")
        echo "    -> ${system} launched (pid=$!, image=${IMAGE}, log=${LOG})"
    done

    echo "    waiting for ${#PIDS[@]} containers..."
    for pid in "${PIDS[@]}"; do
        wait "${pid}" || true
    done

    for entry in "${RC_FILES[@]}"; do
        IFS='|' read -r sys sq rcf poses lg <<< "${entry}"
        RC=$(cat "${rcf}" 2>/dev/null || echo "?")
        if [ -f "${poses}" ]; then
            LINES=$(wc -l < "${poses}")
            echo ">>> OK: ${sys} seq ${sq} — ${LINES} poses (rc=${RC})"
            SUCCESS=$((SUCCESS + 1))
        else
            echo ">>> FAIL: ${sys} seq ${sq} — no output (rc=${RC}, log=${lg})"
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
