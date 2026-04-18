# LiDAR-Inertial SLAM & HD Map Pipeline

> A reproducible LiDAR-inertial SLAM and HD Map benchmark on KITTI and nuScenes. Covers pose-graph optimization, Scan Context loop closure, IMU tight coupling, degeneracy-aware edge sigmas, and Lanelet2 geometry export.

[![CI](https://img.shields.io/github/actions/workflow/status/trouties/lidar-slam-hdmap/ci.yml?branch=main&label=CI)](https://github.com/trouties/lidar-slam-hdmap/actions)
![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Platform](https://img.shields.io/badge/Platform-Linux%20%7C%20WSL2-lightgrey)
![KISS-ICP](https://img.shields.io/badge/KISS--ICP-1.2-brightgreen)
![GTSAM](https://img.shields.io/badge/GTSAM-4.2-orange)
![Open3D](https://img.shields.io/badge/Open3D-0.18-blue)
![Lanelet2](https://img.shields.io/badge/Lanelet2-HD%20Map-blue)

## Pipeline Architecture

```
 Stage 1              Stage 2            Stage 3               Stage 4
 Data Ingestion       LiDAR Odometry     Graph Optimization    Sensor Fusion
┌──────────────┐    ┌───────────────┐   ┌──────────────────┐  ┌─────────────────┐
│ KITTIDataset │    │   KISS-ICP    │   │  GTSAM Pose Graph│  │ Error-State KF  │
│ NuScenesDset │───>│  adaptive ICP │──>│+ Scan Context v2 │─>│ const-velocity  │
│              │    │               │   │  loop closure    │  │ fallback only   │
└──────────────┘    └───────────────┘   └──────────────────┘  └────────┬────────┘
                                                                       │
                     Stage 6              Stage 5              smoothed poses
                     HD Map Export        Mapping              + point clouds
                    ┌───────────────┐   ┌──────────────────┐         │
                    │ Lanelet2 .osm │<──│ Voxel Map Builder│<────────┘
                    │ LineString +  │   │+ Lane/Curb Extr. │
                    │ Area only     │   │                  │
                    └───────────────┘   └──────────────────┘
```

Stage 4 (ESKF) is pose smoothing only; KITTI Odometry has no IMU, so it runs a constant-velocity model. Tight IMU coupling lives in a separate Stage 3.5 branch invoked by `scripts/compare_tight_vs_loose.py` (SUP-04). Stage 6 exports the Lanelet2 geometry layer (`lineStringLayer`); the semantic `laneletLayer` is empty (see Known Limitations FM-3).

## Results

### System Accuracy on KITTI Seq 00

Both first-frame-aligned and SE(3) Umeyama-aligned APE reported (see `src/odometry/kiss_icp_wrapper.py:evaluate_odometry(align=...)`, added 2026-04-18).

| System | APE first (m) | **APE SE(3) (m)** | Paper (m) | Status |
|--------|--------------:|------------------:|----------:|:-------|
| **Ours (fused)** | 10.57 | **4.03** | — | Reproduced |
| FAST-LIO2 | 218.4 | 134.3 | 3–8 | KNOWN_LIMITATION (KITTI deskew incompat) |
| hdl_graph_slam | 215.9 | 200.4 | 6–15 | KNOWN_LIMITATION (LiDAR-only drift) |
| LIO-SAM | 930.0 | 582.5 | 3–7 | KNOWN_LIMITATION (P0-2 structural) |

Our fused pipeline reaches **4.03 m SE(3)-APE** (STRETCH target). The three external baselines were analyzed end-to-end through Phase A/B (see `results/diagnostics/phaseA_report.md` + `phaseB_report.md`); gaps to paper values trace to *independent* non-tunable causes: LIO-SAM's `imuPreintegration.cpp` resets (P0-2 structural), KITTI `.bin` clouds being vendor-deskewed (FAST-LIO2's motion compensation becomes double-compensation), and hdl_graph_slam's LiDAR-only architecture ceiling over a 3 km route. Closing any of them requires source-level baseline patches, not parameter tuning. Full record: `refs/sup-notes.md → SUP-01 Phase A/B Reproduction Attempt`.

### Stage-by-Stage Accuracy on Seq 00

| Configuration | APE RMSE (m) | Delta |
|---------------|-------------:|------:|
| Stage 2: KISS-ICP odometry only | 12.53 | baseline |
| Stage 3: + pose graph + Scan Context | 10.58 | −15.6% |
| Stage 3.5: + IMU tight coupling (SUP-04) | 9.22 | −20.0% vs loose |
| Stage 4: + ESKF pose smoothing | 10.58 | <0.01 m |

### Performance

| Metric | Value |
|--------|------:|
| Stage 2 per-frame latency p50 | 145 ms |
| Stage 2 per-frame latency p95 | 204 ms |
| Full pipeline (200 frames, Seq 00) | 50.6 s |
| Loop closures detected (Seq 00 full) | 2,635 |
| Loop closure precision | 0.967 |
| Loop closure recall | 0.195 |
| Stage 3 speedup (production config, SUP-03) | 2.19× |
| Stage 3 ICP verify speedup (downsample cache) | 3.36× |
| GNSS denial drift (Seq 00, 150 m window) | 0.003 m/m |

### Stage 2 Odometry on nuScenes mini

All 10 scenes pass the Stage 2 APE < 10 m threshold. KISS-ICP adapted for VLP-32C: `voxel_size=0.5`, `min_range=3.0`, 20 Hz sweep mode (2 Hz keyframes cause ICP divergence).

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

Stage 3 loop closure fires zero times on all 10 scenes (mini clips are single-pass segments with no revisits), so Stage 2 and Stage 3 APE are bit-identical.

### Pose Graph Uncertainty Under GNSS Denial (SUP-06)

Per-keyframe marginal covariance from `gtsam.Marginals.jointMarginalCovariance`, rendered as 3D 2σ position ellipsoids. A 354-frame GNSS-denied window (frames 2270–2624) inflates `trace(Σ_pos)` by >26× relative to the non-prior drift baseline, then collapses back within 1.07× as priors resume.

| Mode | drift baseline | denial peak | peak / baseline | post / baseline |
|------|---------------:|------------:|----------------:|----------------:|
| Loose (LiDAR + pose graph) | 0.268 m² | 7.131 m² | **26.61×** | 1.07× |
| Tight (+ IMU, SUP-04) | 0.253 m² | 6.990 m² | **27.59×** | 1.07× |

![SUP-06 loose GNSS denial uncertainty](benchmarks/uncertainty/ellipsoid_animation_00_loose.gif)
![SUP-06 tight (IMU) GNSS denial uncertainty](benchmarks/uncertainty/ellipsoid_animation_00_tight.gif)

Both modes pass acceptance (`peak / baseline ≥ 2×`, `post / baseline ≤ 1.5×`). Reproduce with `python -m scripts.run_sup06 --sequence 00 --mode both`.

### LiDAR Degeneracy Detection (SUP-07)

A 3×3 translation-block Hessian `H_t = Σ nᵢnᵢᵀ` with PCA-normal gating, followed by EMA + min-run hysteresis, flags directionally under-observed frames and downgrades their odometry-edge translation sigmas by 10×. Tuned to separate KITTI Seq 00 (urban) from Seq 01 (highway, LOAM-benchmark degenerate).

![SUP-07 cond_number distribution](benchmarks/sup07/cond_distribution_hist.png)

| Sequence | cond p50 | cond p95 | sustained frames | APE (baseline → downgrade) |
|----------|---------:|---------:|-----------------:|---------------------------:|
| Seq 00 (urban, 4540 f) | 3.07 | 5.51 | 182 (11 runs) | 10.577 → 10.552 m (−0.24%) |
| Seq 01 (highway, 1100 f) | **12.38** | 45.13 | 1080 (3 runs) | 116.80 m unchanged |

![SUP-07 Seq 01 BEV, dense regime](benchmarks/sup07/degeneracy_bev_seq01.png)

Both acceptance criteria pass: (1) Seq 01 `cond_p50` ≥ 2×Seq 00 `cond_p95` (gap 1.12×), (2) APE no-regression. Seq 01's 98% sustained rate reflects that the whole highway IS degenerate, so per-frame downgrade collapses to sequence-level downgrade by design.

## Scope

The complete chain from raw LiDAR scans to Lanelet2 `.osm` output — the open HD-map standard used by Autoware and Apollo. The current export covers the geometry layer (lane / curb polylines and areas); the semantic layer (lanelet relations, left/right pairing, routing topology) is tracked as SUP-12.

The author's geodetic-science background shapes the implementation: explicit WGS84 → UTM (EPSG:32632) reference frames, rigorous Velodyne ↔ camera ↔ world calibration chains, and GTSAM's factor-graph optimization treated as a generalization of least-squares network adjustment.

## Key Features

- **Multi-dataset SLAM** — KITTI HDL-64E and nuScenes VLP-32C with per-dataset parameter adaptation.
- **Scan Context v2 loop closure** — appearance-based place recognition; 2,635 closures on Seq 00 at P=0.967.
- **Tight-coupled IMU preintegration** — GTSAM Forster-2017 factor; −20% APE vs loose fusion on Seq 00.
- **Lanelet2 geometry export** — PCA-classified lane / curb morphology, RDP-simplified, written as `lineStringLayer`.
- **4-system baseline comparison** — Dockerized hdl_graph_slam / FAST-LIO2 / LIO-SAM with APE/RPE tables.
- **Runtime profiling + Stage-3 2.19× speedup** — per-unique-frame downsample cache, zero APE regression.
- **Uncertainty under GNSS denial (SUP-06)** — GTSAM marginals → 3D 2σ ellipsoids, 26–28× inflation then recovery.
- **LiDAR degeneracy detection (SUP-07)** — 3×3 Hessian + hysteresis, per-edge σ downgrade on sustained runs.
- **5-layer deterministic cache** — odometry → features, enabling 15-minute Stage-5 iteration cycles.

## Quick Start

### Docker Compose — one-command reproduction (SUP-09)

Runs the full pipeline on a 200-frame KITTI Seq 00 subset shipped via GitHub Release `sup09-subset-v1` (~278 MB gzipped) and writes `results/ape.txt` + `results/trajectory.png`.

```bash
export SUP09_SUBSET_URL="https://github.com/<user>/lidar-slam-hdmap/releases/download/sup09-subset-v1/kitti_seq00_200.tar.gz"
export SUP09_SUBSET_SHA256="$(curl -sL ${SUP09_SUBSET_URL}.sha256 | awk '{print $1}')"  # optional

docker compose build     # ~5 min
docker compose up        # ~1 min
cat results/ape.txt
```

Reference numbers (200 frames, loose-coupled, v1 loop closure off): **APE RMSE ≈ 2.29 m**, up-phase wall time ≈ 60 s. After the first run the subset is cached in `./cache_sup09/`. Full contract: `refs/sup-notes.md → SUP-09`.

### Docker (generic)

```bash
docker build -t slam-pipeline -f docker/Dockerfile .
docker run -v ~/data/kitti:/data/kitti slam-pipeline --config configs/default.yaml
```

### Native Installation (WSL2 / Linux)

<details>
<summary>Click to expand native setup instructions</summary>

Prerequisites: Ubuntu 22.04, Python 3.10.

```bash
# 1. Virtual environment
python3.10 -m venv ~/slam-env
source ~/slam-env/bin/activate

# 2. Dependencies (numpy MUST stay <2.0 for GTSAM binary compatibility)
pip install "numpy>=1.26,<2.0"
pip install -e ".[dev]"

# 3. Lanelet2 (pick one)
pip install lanelet2     # requires libboost-dev
pip install lanelet2x    # pure Python fallback

# 4. Download KITTI Odometry from https://www.cvlibs.net/datasets/kitti/eval_odometry.php
#    (Velodyne laser data, Calibration files, Ground truth poses 00–10)
#    Extract into ~/data/kitti/odometry/dataset/

# 5. Run
python scripts/verify_kitti.py --root ~/data/kitti/odometry/dataset --sequence 00
python scripts/run_pipeline.py --config configs/default.yaml
python scripts/run_pipeline.py --max-frames 200   # quick test, ~2 min

# 6. Lint and test
ruff check src/ && ruff format --check src/
pytest tests/ -v
```

Store data under WSL2 native `~/data/`, not `/mnt/c/` — cross-filesystem I/O is 10× slower.

</details>

### ROS2 Humble Node (SUP-08)

Real-time Stage 2 + Stage 3 wrapping for RViz2 visualization.

```bash
sudo bash scripts/sup08_install_ros2.sh
bash scripts/sup08_install_ros2.sh --user-part

bash scripts/sup08_acceptance.sh all    # Seq 00 × 500 frames, ~90 s

source /opt/ros/humble/setup.bash
source ~/slam-env-ros2/bin/activate
source ros2_ws/install/setup.bash
ros2 launch lidar_slam_ros2 slam.launch.py sequence:=00 max_frames:=500
```

Three nodes in `ros2_ws/src/lidar_slam_ros2/`: `kitti_player_node`, `odom_node`, `pose_graph_node`.

Acceptance (KITTI Seq 00 × 500 frames):

| Criterion | Measured |
|-----------|----------|
| `colcon build` | PASS — 2 s, 0 warnings |
| `/odom` + `/velodyne_points` in RViz | ✓ |
| Per-frame latency | p50=148 ms, p95=220 ms, max=310 ms (< 500 ms) |
| 500 frames no crash | ✓ |
| APE vs GT | RMSE=5.38 m, RPE=0.04 m |

## Repository Structure

```
lidar-slam-hdmap/
├── src/
│   ├── data/                # Stage 1 — KITTI, nuScenes, IMU loaders
│   ├── odometry/            # Stage 2 — KISS-ICP wrapper
│   ├── optimization/        # Stage 3 — GTSAM, Scan Context, IMU factor
│   ├── fusion/              # Stage 4 — Error-State KF
│   ├── mapping/             # Stage 5 — Voxel map + lane/curb extraction
│   ├── export/              # Stage 6 — Lanelet2 OSM export
│   ├── visualization/       # Trajectory plots, ellipsoid animation
│   ├── benchmarks/          # Evaluator, timing, GNSS denial
│   └── cache/               # 5-layer deterministic cache
├── scripts/                 # Entry points, SUP-0x eval scripts
├── tests/                   # 11 pytest modules
├── configs/default.yaml     # All pipeline parameters
├── benchmarks/              # CSV outputs, runtime profiles, manifest
├── external/                # Dockerized baselines (LIO-SAM, FAST-LIO2, hdl_graph_slam)
├── docker/                  # Dockerfiles
├── ros2_ws/                 # SUP-08 ROS2 package
└── refs/                    # Pipeline notes, backlog, tuning history
```

## Pipeline Stages

| # | Stage | Input → Output | Key decision |
|---|-------|----------------|--------------|
| 1 | **Data Ingestion** | KITTI / nuScenes → `(N, 4)` ndarrays + GT poses | Processing in Velodyne frame; camera-frame conversion at evaluation only. |
| 2 | **LiDAR Odometry** | points → SE(3) odometry | KISS-ICP adaptive threshold; dataset-specific `voxel_size` / `min_range`. |
| 3 | **Graph Optimization** | odometry + clouds + IMU → optimized SE(3) | GTSAM LM + Scan Context v2 (ICP fitness ≥ 0.9), optional IMU factor, optional SUP-07 edge σ downgrade. |
| 4 | **Sensor Fusion** | optimized poses → smoothed SE(3) | ESKF pose smoothing with constant-velocity model when no IMU. SUP-04 tight IMU is a separate script. |
| 5 | **Mapping & Features** | fused poses + clouds → lane / curb clusters + global map | NumPy streaming voxel aggregation. Lane: `intensity ≥ 0.40` + DBSCAN. Curb: height-jump in 0.30 m grid. |
| 6 | **HD Map Export** | clusters → Lanelet2 `.osm` | RDP polyline simplification (ε=0.05 m), separate lane and curb pipelines. `laneletLayer` empty (see FM-3). |

### Supplement Tasks

Completed P0+P1: SUP-01..09 (baseline comparison, Scan Context loop closure, runtime profiling, IMU tight coupling, nuScenes evaluation, uncertainty visualization, degeneracy detection, ROS2 wrapping, Docker Compose reproduction). Pending: SUP-10 Failure Modes section, SUP-11 OSM alignment. Backlog: `refs/backlog.md`.

## Benchmark Report

Every benchmark run is tracked in [`benchmarks/benchmark_manifest.json`](benchmarks/benchmark_manifest.json) with `run_id`, `git_sha`, `config_hash`, `timestamp`, `label`, and artifact paths.

| File | Content |
|------|---------|
| [`benchmarks/accuracy_table.csv`](benchmarks/accuracy_table.csv) | APE/RPE across 4 systems × 2 sequences |
| [`benchmarks/nuscenes_ape.csv`](benchmarks/nuscenes_ape.csv) | nuScenes 10-scene cross-dataset results |
| [`benchmarks/tight_vs_loose/ape_compare.csv`](benchmarks/tight_vs_loose/ape_compare.csv) | IMU tight vs loose coupling |
| [`benchmarks/robustness_gnss_denied.csv`](benchmarks/robustness_gnss_denied.csv) | GNSS denial drift measurements |
| [`benchmarks/runtime_profile_baseline_200f.csv`](benchmarks/runtime_profile_baseline_200f.csv) | Per-stage latency profile (200 frames) |
| [`benchmarks/uncertainty/sup06_report_00.json`](benchmarks/uncertainty/sup06_report_00.json) | SUP-06 marginal covariance metrics |
| [`benchmarks/uncertainty/marginal_cov_00_loose.csv`](benchmarks/uncertainty/marginal_cov_00_loose.csv) | Per-keyframe 3×3 position marginals |
| [`benchmarks/sup07/degeneracy_summary.csv`](benchmarks/sup07/degeneracy_summary.csv) | SUP-07 per-sequence cond statistics |
| [`benchmarks/sup07/ape_compare.csv`](benchmarks/sup07/ape_compare.csv) | SUP-07 two-pass APE comparison |

## Datasets

| Dataset | Sensor | Sequences | Frames | Purpose |
|---------|--------|-----------|-------:|---------|
| [KITTI Odometry](https://www.cvlibs.net/datasets/kitti/eval_odometry.php) | HDL-64E, 10 Hz | 00–10 (with GT) | 4,541 (Seq 00) | Primary benchmark |
| [nuScenes mini](https://www.nuscenes.org/nuscenes) | VLP-32C, 20 Hz sweeps | 10 scenes | 382–400/scene | Cross-dataset generalization (SUP-05) |
| [MulRan](https://sites.google.com/view/mulran-pr/) | Ouster OS1-64 | — | — | Planned: multi-session loop closure |

## Tech Stack

| Layer | Component | Role |
|-------|-----------|------|
| Perception | [KISS-ICP](https://github.com/PRBonn/kiss-icp) | Adaptive-threshold ICP odometry |
| Perception | [Open3D](http://www.open3d.org/) | ICP verification, point cloud processing |
| Optimization | [GTSAM 4.2](https://gtsam.org/) | Factor graph, Levenberg-Marquardt, IMU preintegration |
| Optimization | Scan Context | Appearance-based loop closure descriptor |
| Fusion | Error-State KF | IMU-less constant-velocity fallback |
| Mapping | NumPy streaming voxel | Memory-safe global map aggregation |
| Mapping | DBSCAN (scikit-learn) | Lane marking + curb boundary clustering |
| Export | [Lanelet2](https://github.com/fzi-forschungszentrum-informatik/Lanelet2) | HD Map standard, OSM XML format |
| Evaluation | [evo](https://github.com/MichaelGrupp/evo) | APE/RPE trajectory metrics |
| Infrastructure | Docker | Reproducible baseline comparison |
| CI | GitHub Actions | Ruff lint + pytest |

## Known Limitations

| # | Mode | Root cause & fix path |
|---|------|----------------------|
| FM-1 | ESKF adds no value on KITTI Odometry (Stage 4 APE ≈ Stage 3) | KITTI Odometry has no IMU; Stage 4 runs a constant-velocity model. True IMU integration requires the Stage 3.5 tight-coupled branch (SUP-04). |
| FM-2 | Loop closure recall capped at ~20% (P=0.967, R=0.195) | `max_matches_per_query=5` caps at ~22% structurally; ICP@0.9 prunes further. Raising the cap yields diminishing returns with linear ICP cost. |
| FM-3 | Empty `laneletLayer` — the `.osm` is geometric, not a semantic HD map | Stage 6 writes only `lineStringLayer`. Consumers expecting lane pairing or routing see zero lanelets. Fix: curb-driven pairing, tracked as P1-2 / SUP-12. |
| FM-4 | Flat-ground assumption (lane / curb lost on hills, multi-level) | Fixed `road_z_min/max = [-2.0, -1.5]` window. Fix: per-frame terrain adaptation. P3 task. |
| FM-5 | Conservative IMU noise lock (`accel_noise_sigma=5.0`, ~17× OxTS datasheet) | Workaround for (a) approximate LiDAR↔IMU timestamp alignment, (b) OxTS filtered nav data vs raw IMU mismatch, (c) calibration residuals. Tightening to σ=0.3 inflates Tight APE to 27.85 m. Fix tracked as P0-2. |
| FM-6 | No traffic sign / signal extraction | Stage 5 filters the road-plane z-band only. Fix: SUP-17 heuristic stop-line / crosswalk detection. |
| FM-7 | Seq 01 has zero loop closures in production config | Highway has no revisits — Scan Context cannot fire. SUP-07 downgrade is designed to hand position work to IMU/GNSS priors instead. |

## License

This project is licensed under the [MIT License](LICENSE). Third-party dependencies carry their own licenses (notably `evo` is GPL-3.0, baselines in `external/` are GPL-2.0); these are runtime dependencies or Docker-isolated.

## Acknowledgments

- [KISS-ICP](https://github.com/PRBonn/kiss-icp) — Vizzo et al., RAL 2023
- [GTSAM](https://gtsam.org/) — Dellaert & Kaess, Georgia Tech
- [Open3D](http://www.open3d.org/) — Zhou et al., arXiv 2018
- [Lanelet2](https://github.com/fzi-forschungszentrum-informatik/Lanelet2) — Poggenhans et al., IV 2018
- [evo](https://github.com/MichaelGrupp/evo) — Grupp, TUM
- [nuScenes](https://www.nuscenes.org/) — Caesar et al., CVPR 2020
- [Scan Context](https://github.com/irapkaist/scancontext) — Kim & Kim, IROS 2018
