# LiDAR SLAM HD Map Pipeline

> End-to-end LiDAR-inertial SLAM pipeline with HD Map feature extraction — from raw point clouds to Lanelet2 maps.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![KISS-ICP](https://img.shields.io/badge/KISS--ICP-1.2-green)
![GTSAM](https://img.shields.io/badge/GTSAM-4.2-orange)
![Status](https://img.shields.io/badge/Status-WIP-yellow)

> **⚠️ Work in Progress** — This project is under active development.

## Architecture

```
Raw LiDAR Scans
      │
      ▼
┌─────────────┐    ┌──────────────┐    ┌────────────────┐
│  KISS-ICP   │───▶│  Pose Graph  │───▶│  Global Map    │
│  Odometry   │    │  (GTSAM)     │    │  Construction  │
└─────────────┘    └──────────────┘    └────────────────┘
      │                   │                     │
      ▼                   ▼                     ▼
┌─────────────┐    ┌──────────────┐    ┌────────────────┐
│  IMU/GPS    │    │ Loop Closure │    │   Feature      │
│  Fusion     │    │  Detection   │    │   Extraction   │
│  (ESKF)     │    │              │    │                │
└─────────────┘    └──────────────┘    └────────────────┘
                                              │
                                              ▼
                                       ┌────────────────┐
                                       │  Lanelet2 HD   │
                                       │  Map Export    │
                                       └────────────────┘
```

## Pipeline Stages

| Stage | Module | Description |
|-------|--------|-------------|
| 1 | `src/data/` | KITTI data loading, coordinate transforms |
| 2 | `src/odometry/` | LiDAR odometry via KISS-ICP |
| 3 | `src/optimization/` | Pose graph optimization with GTSAM + loop closure |
| 4 | `src/fusion/` | IMU/LiDAR fusion using Error-State Kalman Filter |
| 5 | `src/mapping/` | Point cloud map construction + feature extraction |
| 6 | `src/export/` | Lanelet2 HD Map export |

## Quick Start

### Docker
```bash
docker build -t slam-pipeline -f docker/Dockerfile .
docker run slam-pipeline --config configs/default.yaml
```

### Manual
```bash
# Activate virtual environment
source ~/slam-env/bin/activate

# Install the package
pip install -e ".[dev]"

# Run the pipeline
python scripts/run_pipeline.py --config configs/default.yaml
```

## Data

This pipeline uses the [KITTI Odometry Dataset](https://www.cvlibs.net/datasets/kitti/eval_odometry.php). Place data at:
```
~/data/kitti/odometry/dataset/
├── sequences/
│   ├── 00/
│   │   ├── velodyne/
│   │   ├── calib.txt
│   │   └── ...
```

## Tech Stack

- **KISS-ICP** — Point-to-point ICP odometry
- **GTSAM** — Factor graph optimization
- **Open3D** — Point cloud processing and visualization
- **Lanelet2** — HD Map format
- **evo** — Trajectory evaluation (APE/RPE)
- **FilterPy** — Kalman filter implementation
