# LiDAR-Inertial SLAM & HD Map Pipeline

> A production-grade LiDAR-inertial SLAM and HD Map feature extraction pipeline on KITTI/nuScenes — with EKF sensor fusion, Scan Context loop closure, and Lanelet2 export — built by a geodesist for autonomous driving localization.

[![CI](https://img.shields.io/github/actions/workflow/status/trouties/lidar-slam-hdmap/ci.yml?branch=main&label=CI)](https://github.com/trouties/lidar-slam-hdmap/actions)
![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Platform](https://img.shields.io/badge/Platform-Linux%20%7C%20WSL2-lightgrey)
![KISS-ICP](https://img.shields.io/badge/KISS--ICP-1.2-brightgreen)
![GTSAM](https://img.shields.io/badge/GTSAM-4.2-orange)
![Open3D](https://img.shields.io/badge/Open3D-0.18-blue)
![Lanelet2](https://img.shields.io/badge/Lanelet2-HD%20Map-blue)

## Pipeline Architecture

```mermaid
graph LR
    subgraph S1["Stage 1: Data Ingestion"]
        A["KITTIDataset\nNuScenesDataset"]
    end
    subgraph S2["Stage 2: LiDAR Odometry"]
        B["KISS-ICP\nadaptive threshold ICP"]
    end
    subgraph S3["Stage 3: Graph Optimization"]
        C["GTSAM Pose Graph\n+ Scan Context v2\n+ IMU Preintegration"]
    end
    subgraph S4["Stage 4: Sensor Fusion"]
        D["Error-State KF\n+ GTSAM tight coupling"]
    end
    subgraph S5["Stage 5: Mapping"]
        E["Voxel Map Builder\n+ Lane/Curb Extraction"]
    end
    subgraph S6["Stage 6: HD Map Export"]
        F["Lanelet2 .osm\n+ GeoJSON features"]
    end

    A -->|"(N,4) float32\npoint clouds"| B
    B -->|"SE(3) 4×4\nodometry poses"| C
    C -->|"optimized\nSE(3) poses"| D
    D -->|"fused poses\n+ point clouds"| E
    E -->|"PCA-classified\nlane/curb clusters"| F
```

## Results

### System Accuracy Comparison

Evaluated with [evo](https://github.com/MichaelGrupp/evo) APE (Absolute Pose Error). Lower is better.

| System | Seq 00 APE RMSE (m) | Seq 00 APE Mean (m) | Seq 05 APE RMSE (m) | Seq 05 APE Mean (m) |
|--------|---------------------:|--------------------:|---------------------:|--------------------:|
| **Ours (fused)** | **11.53** | **10.22** | **3.23** | **2.80** |
| hdl_graph_slam | 78.46 | 68.05 | 56.48 | 33.97 |
| FAST-LIO2 | 77.41 | 61.32 | 20.69 | 15.56 |
| LIO-SAM | 552.85 | 506.00 | 968.82 | 891.86 |

> Baselines run in Docker containers on identical KITTI sequences. See [`external/`](external/) for reproduction scripts.

<!-- INSERT: results/trajectory_comparison_seq00.png -->

### Stage-by-Stage Accuracy Improvement (Seq 00)

| Pipeline Configuration | APE RMSE (m) | Delta |
|------------------------|-------------:|------:|
| Stage 2: KISS-ICP odometry only | 12.53 | baseline |
| Stage 3: + pose graph + Scan Context loop closure | 11.53 | −8.0% |
| Stage 3†: + IMU tight coupling (GTSAM preintegration) | 9.22 | −20.0% vs loose |
| Stage 4: + ESKF fusion | 11.53 | <0.01 m ‡ |

> † Uses KITTI Raw OxTS data via SUP-04 tight coupling path (Forster 2017 IJRR preintegration factor).
>
> ‡ KITTI Odometry contains no IMU data — ESKF uses a constant-velocity model and cannot improve already-optimized poses. Full ESKF value appears on datasets with raw IMU (nuScenes, KITTI Raw).

<!-- INSERT: results/trajectory_seq00_optimized.png -->

### Performance

| Metric | Value |
|--------|------:|
| Stage 2 per-frame latency p50 | 145 ms |
| Stage 2 per-frame latency p95 | 204 ms |
| Full pipeline (200 frames, Seq 00) | 50.6 s |
| Loop closures detected (Seq 00 full) | 2,635 |
| Loop closure precision | 0.967 |
| Loop closure recall | 0.195 |
| Stage 3 optimization speedup (SUP-02 → SUP-03) | 4.77× |
| GNSS denial drift (Seq 00, 150 m window) | 0.003 m/m |

### Cross-Dataset Validation (nuScenes)

All 10 nuScenes mini scenes pass the APE < 10 m acceptance threshold.

| Scene | Frames | Stage 2 APE Mean (m) | Stage 3 APE Mean (m) |
|-------|-------:|---------------------:|---------------------:|
| scene-0553 | 398 | 0.014 | 0.014 |
| scene-0757 | 397 | 0.530 | 0.530 |
| scene-0061 | 382 | 0.698 | 0.698 |
| scene-0103 | 389 | 0.801 | 0.801 |
| scene-0916 | 399 | 0.997 | 0.997 |
| scene-0655 | 396 | 1.908 | 1.908 |
| scene-1094 | 391 | 1.892 | 1.892 |
| scene-0796 | 392 | 2.755 | 2.755 |
| scene-1077 | 400 | 6.730 | 6.730 |
| scene-1100 | 391 | 0.070 | 0.070 |

> KISS-ICP adapted for nuScenes 32-beam VLP-32C: `voxel_size=0.5` (vs KITTI 1.0), `min_range=3.0` (vs 5.0), 20 Hz sweep mode (2 Hz keyframes cause ICP divergence).

<!-- INSERT: results/nuscenes_ape_bar_chart.png -->

## Why This Matters

**HD Maps are L3+ infrastructure.** Level 3 and above autonomous driving systems depend on centimeter-accurate HD maps for lane-level localization, path planning, and regulatory compliance. This pipeline demonstrates the complete chain from raw LiDAR scans to Lanelet2 — the open standard used by Autoware, Apollo, and European OEMs — covering localization, mapping, and map-layer extraction in a single reproducible workflow.

**Geodesy perspective.** The author's background in geodetic science shapes this pipeline differently from pure robotics approaches. Coordinate reference systems are explicit (WGS84 → UTM via EPSG:32632), the Velodyne-to-camera-to-world transformation chain is rigorously managed through calibration matrices, and GTSAM's Gauss-Newton factor graph optimization is structurally identical to geodetic least-squares network adjustment — a technique geodesists have refined for two centuries.

**Scope.** This is not a perception-only project (3D object detection, lane segmentation). It covers the full localization → mapping → HD map export chain that feeds downstream planning and control modules — the infrastructure layer that most portfolio projects skip.

## Key Features

- **Multi-dataset SLAM** — KITTI Odometry (HDL-64E, 64-beam, 10 Hz) and nuScenes mini (VLP-32C, 32-beam, 20 Hz) with dataset-specific parameter adaptation. Not KITTI-overfit.

- **Scan Context v2 loop closure** — Appearance-based place recognition without GPS prior. 2,635 verified closures on Seq 00 at precision 0.967. ICP fitness gate at 0.9 ensures geometric consistency.

- **Tightly-coupled IMU preintegration** — GTSAM Forster 2017 preintegration factor alongside loosely-coupled ESKF. 20% APE reduction on KITTI Seq 00 (11.53 m → 9.22 m).

- **Lanelet2 HD Map export** — Lane markings (intensity ≥ 0.40) and curb boundaries (height-jump detection) extracted, PCA-classified into thin/thick/area morphology, RDP-simplified (ε = 0.05 m), and exported as Lanelet2 `.osm`.

- **4-system baseline comparison** — Benchmarked against hdl_graph_slam, FAST-LIO2, and LIO-SAM in reproducible Docker containers. Full APE/RPE tables on KITTI Seq 00 and 05.

- **5-layer deterministic cache** — Staged caching (odometry → optimized → fused → master map → features) enables 15-minute Stage 5 iteration cycles versus 2 h 20 min cold runs.

- **Per-frame runtime profiling** — p50/p95/max latency distributions per stage. Identified Stage 3 as top bottleneck (57.4%) and achieved 4.77× speedup through Scan Context parameter optimization.

- **GNSS denial resilience** — Drift rate 0.003 m/m over 150 m masked GNSS windows (Seq 00: 0.41 m total drift over 150.3 m denial segment).

## Quick Start

### Docker

```bash
# Build and run on KITTI Seq 00
docker build -t slam-pipeline -f docker/Dockerfile .
docker run -v ~/data/kitti:/data/kitti slam-pipeline --config configs/default.yaml
```

<!-- TBD: SUP-09 will add `docker compose up` one-command reproduction with bundled 200-frame subset -->

### Native Installation (WSL2 / Linux)

<details>
<summary>Click to expand native setup instructions</summary>

#### Prerequisites

- Ubuntu 22.04 (native or WSL2)
- Python 3.10

#### 1. Create virtual environment

```bash
python3.10 -m venv ~/slam-env
source ~/slam-env/bin/activate
```

#### 2. Install dependencies

```bash
# numpy MUST be installed first and stay <2.0 (GTSAM binary compatibility)
pip install "numpy>=1.26,<2.0"
pip install -e ".[dev]"
```

#### 3. Install Lanelet2

```bash
# Option A: Official lanelet2 (requires libboost-dev)
pip install lanelet2

# Option B: lanelet2x (pure Python, cross-platform fallback)
pip install lanelet2x
```

#### 4. Download KITTI Odometry

1. Register at [cvlibs.net](https://www.cvlibs.net/datasets/kitti/user_register.php)
2. Download from the [odometry evaluation page](https://www.cvlibs.net/datasets/kitti/eval_odometry.php):
   - **Velodyne laser data** (80 GB uncompressed)
   - **Calibration files** (1 MB)
   - **Ground truth poses** (4 KB, sequences 00–10 only)
3. Extract into `~/data/kitti/odometry/dataset/`:

```
~/data/kitti/odometry/dataset/
├── sequences/
│   ├── 00/
│   │   ├── velodyne/       # .bin point cloud files
│   │   ├── calib.txt
│   │   └── times.txt
│   └── ...
└── poses/
    ├── 00.txt              # ground truth (seq 00–10)
    └── ...
```

> **WSL2 users**: Store data under `~/data/`, **not** `/mnt/c/` — cross-filesystem I/O is 10× slower.

#### 5. Verify and run

```bash
# Verify KITTI data integrity
python scripts/verify_kitti.py --root ~/data/kitti/odometry/dataset --sequence 00

# Run the full pipeline
python scripts/run_pipeline.py --config configs/default.yaml

# Quick test (200 frames, ~2 min)
python scripts/run_pipeline.py --max-frames 200

# Lint and test
ruff check src/ && ruff format --check src/
pytest tests/ -v
```

</details>

## Repository Structure

```
lidar-slam-hdmap/
├── src/
│   ├── data/                # Stage 1 — KITTI, nuScenes, IMU loaders + coordinate transforms
│   ├── odometry/            # Stage 2 — KISS-ICP wrapper with per-frame timing
│   ├── optimization/        # Stage 3 — GTSAM pose graph, Scan Context, loop closure, IMU factor
│   ├── fusion/              # Stage 4 — Error-State Kalman Filter
│   ├── mapping/             # Stage 5 — Streaming voxel map builder + lane/curb feature extraction
│   ├── export/              # Stage 6 — Lanelet2 OSM export with PCA classification
│   ├── visualization/       # Trajectory plots + point cloud rendering
│   ├── benchmarks/          # Evaluator, timing, GNSS denial, benchmark manifest
│   └── cache/               # 5-layer deterministic cache (odometry → features)
├── scripts/
│   ├── run_pipeline.py      # Main entry point (all 6 stages)
│   ├── benchmark_stage5.py  # 11-sequence Stage 5 iteration benchmarking
│   ├── compare_tight_vs_loose.py  # SUP-04 IMU coupling comparison
│   ├── eval_nuscenes.py     # SUP-05 cross-dataset evaluation
│   ├── profile_stages.py    # SUP-03 per-frame latency profiling
│   └── run_baseline_compare.py    # SUP-01 4-system comparison
├── tests/                   # 11 test modules (pytest)
├── configs/default.yaml     # All pipeline parameters with tuning rationale
├── benchmarks/              # CSV outputs, runtime profiles, benchmark manifest
├── external/                # Dockerized baselines (LIO-SAM, FAST-LIO2, hdl_graph_slam)
├── docker/Dockerfile
├── .github/workflows/ci.yml # Ruff lint + pytest
└── refs/                    # Specs, pipeline notes, tuning history
```

## Pipeline Details

### Stage 1: Data Ingestion

Loads raw sensor data and produces a unified per-frame interface.

- **Input**: KITTI `.bin` point clouds + `calib.txt` + `times.txt` + `poses/*.txt`; or nuScenes database API (20 Hz sweeps)
- **Output**: `(N, 4) float32` ndarrays `[x, y, z, intensity]` + `4×4 float64` GT poses + `float64` timestamps
- **Key decision**: All processing in Velodyne frame (x-forward, y-left, z-up). Camera frame conversion only at evaluation/export boundary.

### Stage 2: LiDAR Odometry

Per-frame point cloud registration via KISS-ICP adaptive-threshold ICP.

- **Input**: per-frame `(N, 4)` point clouds (reflectance stripped to `(N, 3)` for KISS-ICP)
- **Output**: `List[np.ndarray]` of `4×4` SE(3) odometry poses
- **Key decision**: nuScenes uses `voxel_size=0.5` (vs KITTI `1.0`) and `min_range=3.0` (vs `5.0`) to compensate for 32-beam sparsity.

### Stage 3: Graph Optimization

GTSAM factor graph with three constraint types: odometry, GPS prior, and loop closure.

- **Input**: odometry poses + point clouds (for Scan Context + ICP verification) + optional KITTI Raw OxTS IMU data
- **Output**: globally optimized SE(3) poses
- **Key decision**: `icp_fitness_threshold=0.9` — Scan Context is a coarse recall-oriented filter; ICP at 0.9 is the precision gate. This yields P=0.967 at R=0.195 on Seq 00.

### Stage 4: Sensor Fusion

Dual-path fusion: ESKF (loosely-coupled fallback) and GTSAM tight coupling (when IMU available).

- **Input**: optimized poses + optional IMU measurements
- **Output**: fused SE(3) poses
- **Key decision**: KITTI Odometry has no IMU — ESKF uses constant-velocity model (APE delta < 0.01 m). True fusion value appears with KITTI Raw OxTS or nuScenes IMU. Tight coupling via GTSAM preintegration yields −20% APE when IMU is available.

### Stage 5: Mapping & Feature Extraction

Builds a globally consistent point cloud map and extracts road features.

- **Input**: fused poses + per-frame point clouds (cached master map at `voxel_size=0.10`)
- **Output**: lane clusters + curb clusters (GeoJSON) + global map (`.pcd`)
- **Key decision**: numpy-native streaming voxel aggregation (not Open3D accumulate) to stay under 4 GB RAM. Lane detection via `intensity ≥ 0.40` + DBSCAN (`eps=0.7`, `min_points=40`). Curb detection via height-jump analysis in 0.30 m grid cells.

### Stage 6: HD Map Export

Converts classified feature clusters to Lanelet2 OSM format.

- **Input**: PCA-classified lane/curb clusters (thin/thick/area morphology)
- **Output**: Lanelet2 `.osm` with LineString ways + Area relations + geometry metadata tags (`length_m`, `thickness_m`, `linearity`)
- **Key decision**: RDP polyline simplification at ε = 0.05 m reduces OSM size by 7%. Separate classification pipelines for lane (3-class) and curb (single class with rescue trim at `trim_k=1.0`).

### Supplement Tasks

| ID | Task | Priority | Status | Key Result |
|----|------|----------|--------|------------|
| SUP-01 | 4-system baseline comparison | P0 | Done | Ours 11.53 m vs next-best 77.41 m (Seq 00) |
| SUP-02 | Scan Context v2 loop closure | P0 | Done | 2,635 closures, P=0.967, R=0.195 |
| SUP-03 | Runtime profiling report | P0 | Done | Stage 3 bottleneck identified, 4.77× speedup |
| SUP-04 | IMU preintegration tight coupling | P0 | Done | APE −20% (11.53 → 9.22 m) |
| SUP-05 | nuScenes cross-dataset evaluation | P1 | Done | 10 scenes, all APE < 10 m |
| SUP-06 | Uncertainty visualization | P1 | Planned | Covariance ellipsoids from pose graph marginals |
| SUP-07 | Degeneracy detection | P1 | Planned | ICP Hessian condition number monitoring |
| SUP-08 | ROS2 Humble node wrapping | P1 | Planned | Real-time /odom + TF + RViz2 |
| SUP-09 | Docker Compose reproduction | P1 | Planned | `docker compose up` one-command demo |
| SUP-10 | Failure modes documentation | P1 | Planned | ≥5 modes with visual evidence |
| SUP-11 | OSM alignment evaluation | P1 | Planned | Hausdorff distance vs OSM road network |
| SUP-12 | Lanelet2 routing graph + A* | P2 | Planned | Path query on HD Map topology |
| SUP-13 | Fixed-lag smoother vs full batch | P2 | Planned | iSAM2 vs LM trade-off analysis |
| SUP-14 | Interactive web demo (Kepler.gl) | P2 | Planned | GitHub Pages deployment |
| SUP-15 | Technical writeup | P2 | Planned | 2,500–3,500 word engineering report |
| SUP-16 | Demo video | P2 | Planned | ≤30 s trajectory replay animation |
| SUP-17 | HD Map semantic layer | P2 | Planned | Heuristic stop line + crosswalk detection |

## Benchmark Report

Every benchmark run is tracked in [`benchmarks/benchmark_manifest.json`](benchmarks/benchmark_manifest.json) with fields: `run_id`, `git_sha`, `config_hash`, `timestamp`, `label`, and artifact paths.

Key data files:

| File | Content |
|------|---------|
| [`benchmarks/accuracy_table.csv`](benchmarks/accuracy_table.csv) | APE/RPE across 4 systems × 2 sequences |
| [`benchmarks/nuscenes_ape.csv`](benchmarks/nuscenes_ape.csv) | nuScenes 10-scene cross-dataset results |
| [`benchmarks/tight_vs_loose/ape_compare.csv`](benchmarks/tight_vs_loose/ape_compare.csv) | IMU tight vs loose coupling comparison |
| [`benchmarks/robustness_gnss_denied.csv`](benchmarks/robustness_gnss_denied.csv) | GNSS denial drift measurements |
| [`benchmarks/runtime_profile_baseline_200f.csv`](benchmarks/runtime_profile_baseline_200f.csv) | Per-stage latency profile (200 frames) |

## Datasets

| Dataset | Sensor | Sequences | Frames | Purpose |
|---------|--------|-----------|-------:|---------|
| [KITTI Odometry](https://www.cvlibs.net/datasets/kitti/eval_odometry.php) | HDL-64E (64-beam, 10 Hz) | 00–10 (with GT) | 4,541 (Seq 00) | Primary benchmark, all stages |
| [nuScenes mini](https://www.nuscenes.org/nuscenes) | VLP-32C (32-beam, 20 Hz sweeps) | 10 scenes | 382–400/scene | Cross-dataset generalization (SUP-05) |
| [MulRan](https://sites.google.com/view/mulran-pr/) | Ouster OS1-64 | — | — | Planned: multi-session loop closure |

## Tech Stack

| Layer | Component | Role |
|-------|-----------|------|
| **Perception** | [KISS-ICP](https://github.com/PRBonn/kiss-icp) | Adaptive-threshold point-to-point ICP odometry |
| **Perception** | [Open3D](http://www.open3d.org/) | ICP verification, point cloud processing, visualization |
| **Optimization** | [GTSAM 4.2](https://gtsam.org/) | Factor graph, Levenberg-Marquardt, IMU preintegration |
| **Optimization** | Scan Context | Appearance-based loop closure descriptor (self-implemented) |
| **Fusion** | Error-State KF | IMU-less constant-velocity fallback (self-implemented) |
| **Mapping** | NumPy streaming voxel | Memory-safe global map aggregation (<4 GB) |
| **Mapping** | DBSCAN (scikit-learn) | Lane marking + curb boundary clustering |
| **Export** | [Lanelet2](https://github.com/fzi-forschungszentrum-informatik/Lanelet2) | HD Map standard, OSM XML format |
| **Evaluation** | [evo](https://github.com/MichaelGrupp/evo) | APE/RPE trajectory metrics |
| **Infrastructure** | Docker | Reproducible baseline comparison containers |
| **CI** | GitHub Actions | Ruff lint + pytest on every push |

## Known Limitations & Failure Modes

### FM-1: ESKF adds no value on KITTI Odometry

- **Symptom**: Stage 4 APE = Stage 3 APE (delta < 0.01 m)
- **Root cause**: KITTI Odometry has no IMU data. ESKF uses constant-velocity model, which cannot improve already-optimized poses.
- **Status**: By design. Use KITTI Raw or nuScenes for full ESKF evaluation. Tight coupling (SUP-04) demonstrates real IMU benefit (−20% APE).

### FM-2: Loop closure recall capped at ~20%

- **Symptom**: Recall 0.195 despite 2,635 closures at precision 0.967
- **Root cause**: Ground truth defines ~13,050 valid pairs. `max_matches_per_query=5` creates a structural ceiling of ~22%. ICP fitness threshold at 0.9 further prunes.
- **Status**: Dataset-inherent limitation. Increasing `max_matches_per_query` yields diminishing returns with linear ICP cost growth.

### FM-3: Empty lanelet relations

- **Symptom**: Lanelet2 `.osm` contains LineStrings and Areas but no `<relation>` elements for drivable lanes.
- **Root cause**: Curb-driven left/right lane boundary pairing not yet implemented.
- **Status**: P2 task (pipeline-notes §12). Prerequisite for SUP-12 routing graph.

### FM-4: Flat-ground assumption in feature extraction

- **Symptom**: Lane markings and curbs lost on hills or multi-level road structures.
- **Root cause**: `road_z_min/max = [-2.0, -1.5]` is a fixed window relative to Velodyne height (1.73 m). No per-frame terrain adaptation.
- **Status**: P3 task. Requires ground plane estimation per segment.

### FM-5: Conservative IMU noise parameters

- **Symptom**: `accel_noise_sigma=5.0` is ~100× looser than OxTS datasheet values.
- **Root cause**: KITTI OxTS outputs filtered navigation data (not raw IMU). Tighter parameters (σ = 0.3) cause APE to explode to 27.85 m (+154%) due to timestamp alignment approximation and calibration residuals.
- **Status**: Locked. Solving requires precise KITTI Raw timestamp synchronization or a dataset with true raw IMU output.

### FM-6: No traffic sign or signal extraction

- **Symptom**: HD map contains only ground-level features (lane markings, curbs).
- **Root cause**: Stage 5 filters to road-plane z-band only. Vertical structures are excluded by design.
- **Status**: SUP-17 (P2) will add heuristic stop line and crosswalk detection.

<!-- TBD: SUP-10 will add visual evidence (screenshots, trajectory overlays) for each failure mode -->

## Contributing & License

### Contributing

```bash
# Lint and format check (required before PR)
ruff check src/ && ruff format --check src/

# Run tests
pytest tests/ -v
```

See [`.github/workflows/ci.yml`](.github/workflows/ci.yml) for CI checks that run on every push.

### License

This project is licensed under the [MIT License](LICENSE).

> **Third-party dependencies** have their own licenses (notably `evo` is GPL-3.0 and baseline systems in `external/` are GPL-2.0). These are runtime dependencies or Docker-isolated — they do not affect the license of this project's source code.

## Acknowledgments

This project builds on excellent open-source work:

- [KISS-ICP](https://github.com/PRBonn/kiss-icp) — Vizzo et al., RAL 2023
- [GTSAM](https://gtsam.org/) — Dellaert & Kaess, Georgia Tech
- [Open3D](http://www.open3d.org/) — Zhou et al., arXiv 2018
- [Lanelet2](https://github.com/fzi-forschungszentrum-informatik/Lanelet2) — Poggenhans et al., IV 2018
- [evo](https://github.com/MichaelGrupp/evo) — Grupp, TUM
- [nuScenes](https://www.nuscenes.org/) — Caesar et al., CVPR 2020
- [Scan Context](https://github.com/irapkaist/scancontext) — Kim & Kim, IROS 2018

## Author

**Haotian Zha** — Geodesist turned autonomous driving engineer.

This project bridges geodesy and autonomous driving perception. My background in geodetic science — coordinate reference systems, network adjustment, uncertainty propagation — shapes how I approach SLAM differently from a pure robotics perspective. Where most SLAM implementations treat the coordinate frame as a detail, this pipeline explicitly manages the Velodyne-to-camera-to-UTM transformation chain, applies geodetic-grade factor graph optimization (GTSAM's Gauss-Newton is structurally identical to geodetic least-squares network adjustment), and produces output in Lanelet2 — a format designed for regulatory-grade HD maps.

[Email](mailto:haotian.zha@gmail.com) · [LinkedIn](https://www.linkedin.com/in/haotianzha/) · [GitHub](https://github.com/trouties)
