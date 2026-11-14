# Evaluation protocol

Every trajectory number in this repository is one of three metrics. They
answer different questions, and on the same trajectory they can differ by a
factor of two or more, so each table states which one it reports.

| Metric | Question it answers | Alignment before comparison | Used for |
|---|---|---|---|
| KITTI relative error, `t_rel` [%] and `r_rel` [deg/100 m] | How fast does the estimate drift per distance travelled? | None: compares relative motion over 100–800 m sub-trajectories | Checking the odometry stage against published results |
| APE, first-frame | How far is each pose from the truth if the map is anchored at the start pose? | Estimate and reference share the first pose | The error of a map built without any external reference |
| APE, SE(3) | How distorted is the shape of the trajectory? | Least-squares rigid fit (Umeyama, no scale) over all poses | The error that remains once the map is georeferenced |

## The two APE variants are a datum choice

In surveying terms, SE(3)-aligned APE is the residual after fitting a
six-parameter Helmert transformation (rotation and translation, no scale) to
all poses: it removes the datum and keeps only the distortion of the network.
First-frame APE instead fixes the datum at a single station with a fixed
orientation. Any orientation error at that station propagates linearly with
distance from it, exactly like an orientation error at the starting point of
an open traverse. A small heading or pitch error early in a sequence
therefore inflates first-frame APE over the whole trajectory, while SE(3)
alignment absorbs most of it.

Neither variant is "the right one". First-frame APE is what an offline map
anchored at its first scan actually carries; SE(3) APE is what is left after
the map has been tied to an external frame. This repository reports both.

A third datum sits between them and is reported as a diagnostic
(`ape_rot_start`): the start position stays fixed, and the orientation is
the rotation about it that best fits the reference. It has three free
parameters. The drop from first-frame APE to this value is the part of the
error that a single orientation offset explains. The remaining drop down to
SE(3) APE is what the free origin absorbs on top of that.

## KITTI relative error

`src/benchmarks/kitti_metrics.py` is a port of `evaluate_odometry.cpp` from
the KITTI odometry devkit:

- Segments of 100, 200, …, 800 m path length, measured along the ground
  truth, start at every 10th frame.
- For each segment, the error is the relative pose
  `inv(ΔT_est) · ΔT_gt` between its first and last frame; translation error
  and rotation angle are divided by the segment length.
- `t_rel` and `r_rel` are means over all segments. For a set of sequences,
  segments are pooled before averaging, as the devkit does, so long
  sequences weigh more. This is the "00–10 average" that papers report.
- Poses are compared in the left camera frame, the frame of the KITTI
  `poses/XX.txt` files.

The port agrees with the C++ implementation shipped in KISS-ICP
(`kiss_icp.metrics.sequence_error`) to a relative 1e-6 in translation and
about 0.1 % in rotation (`tests/test_kitti_metrics.py`).

Not part of the devkit, but reported alongside it: the mean absolute
translation error per camera axis (lateral x, vertical y, forward z), in % of
segment length. It tells a pitch problem (vertical) from a scale problem
(forward) without looking at plots.

## APE

APE is computed with [evo](https://github.com/MichaelGrupp/evo) on the
translation part, in the Velodyne frame, which is the frame the pipeline
runs in. Converting both trajectories to the camera frame changes APE by less
than 1 cm (measured on sequence 04: 2.9557 m vs 2.9618 m first-frame for the
uncorrected odometry, 0.2409 m vs 0.2409 m SE(3) for KISS-ICP's reference
run), so the choice of frame does not matter at the precision reported here.

## Checking the odometry stage against KISS-ICP

Stage 2 wraps KISS-ICP. To check that the wrapper loses no accuracy, it is
compared on sequences 00–10 with the official KISS-ICP pipeline (release
1.2.3, its own KITTI dataloader and default configuration) run on the same
machine, and with the numbers in the KISS-ICP paper (Vizzo et al., RA-L
2023).

KITTI `t_rel` [%]:

| Sequence | Path [m] | Stage 2 | Official KISS-ICP | Stage 2 without scan correction | Paper |
|---|---:|---:|---:|---:|---:|
| 00 | 3724 | 0.520 | 0.528 | 0.909 | 0.51 |
| 01 | 2453 | 0.784 | 0.786 | 2.108 | |
| 02 | 5067 | 0.532 | 0.537 | 1.210 | |
| 03 | 561 | 0.677 | 0.677 | 0.913 | |
| 04 | 394 | 0.388 | 0.385 | 0.930 | 0.36 |
| 05 | 2206 | 0.345 | 0.342 | 0.772 | |
| 06 | 1233 | 0.282 | 0.281 | 0.646 | |
| 07 | 695 | 0.368 | 0.376 | 0.498 | |
| 08 | 3223 | 0.824 | 0.820 | 1.113 | |
| 09 | 1705 | 0.536 | 0.534 | 0.873 | |
| 10 | 920 | 0.527 | 0.512 | 1.443 | |
| 00–10, segments pooled | 22 180 | 0.5625 | 0.5638 | 1.0540 | |

Pooled `r_rel` is 0.161, 0.160 and 0.403 deg/100 m for the three runs.

- **The wrapper matches the official implementation.** The largest
  per-sequence difference is 0.015 percentage points (sequence 10), and the
  pooled values differ by 0.0013. The two runs differ in one setting,
  `min_range` (5 m here, 0 m in KISS-ICP's default). Neither run deskews:
  KISS-ICP's KITTI dataloader passes no per-point timestamps, which has the
  same effect as `deskew=False` here.
- **Both are above the paper.** The paper reports 0.50 % over 00–10, and
  0.51 % and 0.36 % on sequences 00 and 04. Averaging per sequence instead
  of pooling segments gives 0.526 % (Stage 2) and 0.525 % (official), which
  is still above 0.50 %. The official release shows the same gap on the
  same machine, so the gap lies between the version of KISS-ICP in the paper
  and the current release, not in this wrapper. Its cause has not been
  investigated.
- **The KITTI scan correction is required.** `data.correct_kitti_scan`
  raises every beam's elevation by 0.205°, as KISS-ICP's own KITTI
  dataloader does. Without it, the HDL-64E intrinsic calibration makes flat
  ground appear to drop with range, and each registration absorbs part of
  that as a pitch error. Without the correction, the pooled lateral and
  forward errors barely change (0.32 % and 0.24 %, against 0.32 % and
  0.23 % with it), while the vertical error is three times larger (0.88 %
  against 0.28 %).

The CSV also holds the poses after pose-graph optimisation (`stage3`). They
are discussed in [loop-closure.md](loop-closure.md).

To reproduce:

```bash
for s in 00 01 02 03 04 05 06 07 08 09 10; do
  python scripts/run_pipeline.py --sequence $s --until optimized
done
python -m scripts.eval_kitti_odometry run --variant kiss_icp_reference
python -m scripts.eval_kitti_odometry run --variant uncorrected
python -m scripts.eval_kitti_odometry evaluate --csv benchmarks/kitti_odometry.csv
```

## Why first-frame APE is twice SE(3) APE on sequence 00

Odometry (Stage 2) on sequence 00 under the three datums:

| Datum | Free parameters | KITTI scan correction on | off |
|---|---:|---:|---:|
| First frame: start position and orientation fixed | 0 | 6.75 m (vertical 5.38 m) | 12.53 m (vertical 11.77 m) |
| Start position fixed, orientation free | 3 | 4.39 m | 4.88 m |
| SE(3) Umeyama | 6 | 3.43 m | 4.02 m |
| KITTI `t_rel` (vertical part) | — | 0.52 % (0.22 %) | 0.91 % (0.70 %) |

Most of the first-frame error is vertical: 63 % of its mean squared error
with the scan correction, and 88 % without it. Freeing only the three
orientation parameters at the start removes 71 % of the gap between
first-frame and SE(3) APE, and 90 % without the correction. So the gap
between the two numbers is mainly a tilt of the whole trajectory relative
to the reference, not a difference in shape. A roll or pitch offset turns
into a height error that grows linearly with horizontal distance from the
start: 500 m from the start, a tilt of 0.1° is already 0.87 m. This is the open-traverse effect described above, and it
is why first-frame APE punishes attitude errors much harder than shape
errors.

The scan correction illustrates this. It lowers the vertical drift from
0.70 % to 0.22 % and halves first-frame APE (12.53 → 6.75 m), but SE(3)
APE improves only from 4.02 to 3.43 m. Most of what the correction removes
is the tilt, which SE(3) alignment would have absorbed anyway. A comparison
of two methods under first-frame APE alone is therefore largely a comparison
of how well each one holds its attitude.

To reproduce (the "off" column is the `uncorrected` comparison variant; it
matches a pipeline run with `data.correct_kitti_scan: false` to within
0.1 mm per pose):

```bash
python scripts/run_pipeline.py --sequence 00 --until optimized
python -m scripts.eval_kitti_odometry run --variant uncorrected --sequences 00
python -m scripts.eval_kitti_odometry evaluate --sequences 00
```

## Conventions used in this repository

- The odometry stage is checked with the KITTI relative error (`t_rel`,
  `r_rel`), pooled over sequences 00–10, against the KISS-ICP paper and
  the official KISS-ICP implementation run on the same machine.
- Every APE number states its alignment. Tables give first-frame and SE(3)
  APE side by side, both RMSE of the translation, in the Velodyne frame.
- Numbers come from a cold run (`--force-rebuild all`) or are marked as
  cache hits.
- The KITTI ground truth is a GPS/INS solution and is only as good as its GPS
  fix. On sequence 00, 17 % of the frames have an RTK-integer fix; where two
  passes over the same road meet, the ground truth disagrees with the scans
  by 0.4 m vertically and 0.6 m horizontally (medians). Differences between
  trajectories below that level are not resolved by this reference
  ([loop-closure.md](loop-closure.md#the-reference-is-not-consistent-at-revisits)).
- The KITTI ground-truth positions are in the projection the KITTI devkit
  uses, which is not conformal on the WGS84 ellipsoid: distances along
  north–south and east–west come out with slightly different scales. This
  matters little for `t_rel` and for comparisons between trajectories, and
  much for the SE(3) APE of the best trajectories
  ([ground-truth-projection.md](ground-truth-projection.md)).
