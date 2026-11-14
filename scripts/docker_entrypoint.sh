#!/usr/bin/env bash
# Entrypoint of docker/Dockerfile.
#   1. Without the KITTI subset under /data, download and extract the tarball
#      from $KITTI_SUBSET_URL (checked against $KITTI_SUBSET_SHA256 if set).
#   2. Run the pipeline on the subset with configs/default.yaml.
#   3. Write results/ape.txt and results/trajectory.png (scripts/docker_report.py).
set -euo pipefail

DATA_DIR="${DATA_DIR:-/data}"
RESULTS_DIR="${RESULTS_DIR:-/app/results}"
SEQUENCE="${SEQUENCE:-00}"
MAX_FRAMES="${MAX_FRAMES:-200}"
KITTI_SUBSET_URL="${KITTI_SUBSET_URL:-}"
KITTI_SUBSET_SHA256="${KITTI_SUBSET_SHA256:-}"

mkdir -p "${DATA_DIR}" "${RESULTS_DIR}"
sentinel="${DATA_DIR}/sequences/${SEQUENCE}/velodyne/000000.bin"
if [ ! -f "${sentinel}" ]; then
    if [ -z "${KITTI_SUBSET_URL}" ]; then
        echo "subset missing at ${sentinel} and KITTI_SUBSET_URL is unset" >&2
        exit 2
    fi
    echo "downloading the subset from ${KITTI_SUBSET_URL}"
    tmp_tar="$(mktemp /tmp/kitti_subset.XXXXXX.tar.gz)"
    trap 'rm -f "${tmp_tar}"' EXIT
    curl -L --fail --silent --show-error -o "${tmp_tar}" "${KITTI_SUBSET_URL}"
    if [ -n "${KITTI_SUBSET_SHA256}" ]; then
        actual="$(sha256sum "${tmp_tar}" | awk '{print $1}')"
        if [ "${actual}" != "${KITTI_SUBSET_SHA256}" ]; then
            echo "sha256 mismatch: expected ${KITTI_SUBSET_SHA256}, got ${actual}" >&2
            exit 3
        fi
    fi
    tar -xzf "${tmp_tar}" -C "${DATA_DIR}"
    [ -f "${sentinel}" ] || { echo "tarball extracted but ${sentinel} missing" >&2; exit 4; }
fi

python scripts/run_pipeline.py \
    --config configs/default.yaml \
    --kitti-root "${DATA_DIR}" \
    --output-dir "${RESULTS_DIR}" \
    --sequence "${SEQUENCE}" \
    --max-frames "${MAX_FRAMES}"

python -m scripts.docker_report \
    --results-dir "${RESULTS_DIR}" \
    --sequence "${SEQUENCE}" \
    --dataset-root "${DATA_DIR}"
