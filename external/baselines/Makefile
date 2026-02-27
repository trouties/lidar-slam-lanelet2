# SUP-01 External Baseline Docker Builds
# Usage:
#   make build-all          # Build all Docker images
#   make run-all            # Build + run all baselines on seq 00,05
#   make run-hdl SEQ=00     # Run hdl_graph_slam on seq 00 only

SEQUENCES ?= 00,05
DOCKER_BUILD = docker build
BASEDIR = $(shell pwd)

.PHONY: build-base build-hdl build-fastlio build-liosam build-all
.PHONY: run-hdl run-fastlio run-liosam run-all clean

# ---- Build targets ----

build-base:
	$(DOCKER_BUILD) -t slam-baselines/ros-base:noetic \
		-f common/Dockerfile.ros_base $(BASEDIR)

build-hdl: build-base
	$(DOCKER_BUILD) -t slam-baselines/hdl_graph_slam:latest \
		-f hdl_graph_slam/Dockerfile $(BASEDIR)

build-fastlio: build-base
	$(DOCKER_BUILD) -t slam-baselines/fast_lio2:latest \
		-f fast_lio2/Dockerfile $(BASEDIR)

build-liosam: build-base
	$(DOCKER_BUILD) -t slam-baselines/lio_sam:latest \
		-f lio_sam/Dockerfile $(BASEDIR)

build-all: build-hdl build-fastlio build-liosam

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
	docker rmi slam-baselines/ros-base:noetic 2>/dev/null || true
