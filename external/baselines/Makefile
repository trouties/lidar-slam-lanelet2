# SUP-01 External Baseline Docker Builds
# Usage:
#   make build-all          # Build all Docker images (single lio_sam :latest tag)
#   make build-liosam-00    # Alias for build-liosam — per-seq extrinsic is a runtime concern (Stage C envsubst)
#   make run-all            # Build + run all baselines on seq 00,05
#   make run-hdl SEQUENCES=00     # Run hdl_graph_slam on seq 00 only

SEQUENCES ?= 00,05
DOCKER_BUILD = docker build
BASEDIR = $(shell pwd)
KITTI_RAW_DIR ?= $(HOME)/data/kitti_raw
# Use the project venv so inject_kitti_extrinsic.py can import numpy. Falls
# back to system python3 if the venv is absent (CI / fresh checkout).
PYTHON ?= $(shell test -x $(HOME)/slam-env/bin/python && echo $(HOME)/slam-env/bin/python || echo python3)

.PHONY: build-base build-bagbuilder build-hdl build-fastlio build-liosam build-all
.PHONY: build-liosam-00 build-liosam-05
.PHONY: prepare-bags verify-bag-00 verify-bag-05 run-hdl run-fastlio run-liosam run-all clean

# ---- Build targets ----

build-base:
	$(DOCKER_BUILD) -t slam-baselines/ros-base:noetic \
		-f common/Dockerfile.ros_base $(BASEDIR)

build-bagbuilder: build-base
	$(DOCKER_BUILD) -t slam-baselines/bag-builder:latest \
		-f common/Dockerfile.bag_builder $(BASEDIR)

build-hdl: build-base
	$(DOCKER_BUILD) -t slam-baselines/hdl_graph_slam:latest \
		-f hdl_graph_slam/Dockerfile $(BASEDIR)

build-fastlio: build-base
	$(DOCKER_BUILD) -t slam-baselines/fast_lio2:latest \
		-f fast_lio2/Dockerfile $(BASEDIR)

# LIO-SAM build: single image, per-seq extrinsic rendered at runtime via
# envsubst in run.sh from KITTI Raw calib_imu_to_velo.txt (mounted read-only
# at /data/kitti_raw). Replaces the SUP-01 §20.4 build-time injection that
# suffered from docker-layer-cache divergence. See params.yaml.tpl.
build-liosam: build-base
	$(DOCKER_BUILD) -t slam-baselines/lio_sam:latest \
		-f lio_sam/Dockerfile $(BASEDIR)

# Backwards-compat aliases — a few older scripts still call these names.
# They resolve to the same image (no per-seq tag exists post-Stage C).
build-liosam-00: build-liosam
build-liosam-05: build-liosam

build-all: build-bagbuilder build-hdl build-fastlio build-liosam

# ---- Bag cache preparation ----

prepare-bags: build-bagbuilder
	bash prepare_bags.sh "$(SEQUENCES)"

# ---- Stage B: bag smoke verification ----
# Exit non-zero on any failed invariant (counts / monotonicity / PC2 fields /
# gravity). Runs inside the bag-builder image since host has no rosbag.

KITTI_BAG_CACHE ?= $(HOME)/data/kitti_bags_cache

verify-bag-00: build-bagbuilder
	docker run --rm \
		-v "$(KITTI_BAG_CACHE):/cache:ro" \
		--entrypoint python3 \
		slam-baselines/bag-builder:latest \
		/scripts/verify_bag_impl.py --bag /cache/kitti_00_fixed.bag --sequence 00

verify-bag-05: build-bagbuilder
	docker run --rm \
		-v "$(KITTI_BAG_CACHE):/cache:ro" \
		--entrypoint python3 \
		slam-baselines/bag-builder:latest \
		/scripts/verify_bag_impl.py --bag /cache/kitti_05_fixed.bag --sequence 05

# ---- Run targets ----

run-hdl: build-hdl
	bash run_all_baselines.sh "$(SEQUENCES)" "hdl_graph_slam"

run-fastlio: build-fastlio
	bash run_all_baselines.sh "$(SEQUENCES)" "fast_lio2"

run-liosam: build-liosam
	bash run_all_baselines.sh "$(SEQUENCES)" "lio_sam"

run-all: build-all
	bash run_all_baselines.sh "$(SEQUENCES)" "hdl_graph_slam,fast_lio2,lio_sam"

# ---- Cleanup ----

clean:
	docker rmi slam-baselines/hdl_graph_slam:latest 2>/dev/null || true
	docker rmi slam-baselines/fast_lio2:latest 2>/dev/null || true
	docker rmi slam-baselines/lio_sam:latest 2>/dev/null || true
	docker rmi slam-baselines/bag-builder:latest 2>/dev/null || true
	docker rmi slam-baselines/ros-base:noetic 2>/dev/null || true
