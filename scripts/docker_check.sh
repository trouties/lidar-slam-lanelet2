#!/usr/bin/env bash
# Build and run the Docker reproduction, and record what it produced.
#
#   bash scripts/docker_check.sh local   # subset extracted into ./data_subset beforehand
#   bash scripts/docker_check.sh cold    # empty ./data_subset; the container downloads the
#                                        # tarball, served from the host like a release asset
#
# The tarball comes from scripts/package_kitti_subset.py (default
# /tmp/kitti_seq00_200.tar.gz, override with KITTI_SUBSET_TARBALL).
# Writes benchmarks/docker/run.json: APE of the run, subset and image digests.
set -euo pipefail

MODE="${1:-local}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${REPO_ROOT}/benchmarks/docker"
RESULTS_DIR="${REPO_ROOT}/results/docker"
DATA_DIR="${REPO_ROOT}/data_subset"
TARBALL="${KITTI_SUBSET_TARBALL:-/tmp/kitti_seq00_200.tar.gz}"
mkdir -p "${OUT_DIR}" "${RESULTS_DIR}"
cd "${REPO_ROOT}"

stop_http() { [ -n "${HTTP_PID:-}" ] && kill "${HTTP_PID}" 2>/dev/null || true; }
trap stop_http EXIT

case "${MODE}" in
    local)
        if [ ! -f "${DATA_DIR}/sequences/00/velodyne/000000.bin" ]; then
            [ -f "${TARBALL}" ] || { echo "missing ${TARBALL}" >&2; exit 2; }
            mkdir -p "${DATA_DIR}"
            tar -xzf "${TARBALL}" -C "${DATA_DIR}"
        fi
        unset KITTI_SUBSET_URL KITTI_SUBSET_SHA256 || true
        ;;
    cold)
        [ -f "${TARBALL}" ] || { echo "missing ${TARBALL}" >&2; exit 2; }
        rm -rf "${DATA_DIR}" && mkdir -p "${DATA_DIR}"
        port="${KITTI_SUBSET_PORT:-8098}"
        python3 -m http.server "${port}" --bind 0.0.0.0 --directory "$(dirname "${TARBALL}")" \
            > /dev/null 2>&1 &
        HTTP_PID=$!
        sleep 1
        export KITTI_SUBSET_URL="http://host.docker.internal:${port}/$(basename "${TARBALL}")"
        export KITTI_SUBSET_SHA256="$(sha256sum "${TARBALL}" | awk '{print $1}')"
        ;;
    *) echo "usage: $0 [local|cold]" >&2; exit 2 ;;
esac

rm -f "${RESULTS_DIR}/ape.txt" "${RESULTS_DIR}/trajectory.png"
t0=$(date +%s)
docker compose build
t1=$(date +%s)
docker compose up --abort-on-container-exit --exit-code-from pipeline
t2=$(date +%s)

subset_sha256="unknown"
[ -f "${TARBALL}" ] && subset_sha256="$(sha256sum "${TARBALL}" | awk '{print $1}')"
image_id="$(docker image inspect lidar-slam-lanelet2:latest --format '{{.Id}}')"
python3 - "${RESULTS_DIR}/ape.txt" "${OUT_DIR}/run.json" <<PY
import json, sys
ape = dict(line.split("=", 1) for line in open(sys.argv[1]).read().split())
run = {
    "mode": "${MODE}",
    "image_id": "${image_id}",
    "subset_sha256": "${subset_sha256}",
    "build_seconds": $((t1 - t0)),
    "run_seconds": $((t2 - t1)),
    **{k: float(v) if "ape" in k else int(v) for k, v in ape.items()},
}
json.dump(run, open(sys.argv[2], "w"), indent=2)
print(json.dumps(run, indent=2))
PY
