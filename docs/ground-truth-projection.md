# The KITTI ground truth is not metric

The KITTI odometry ground truth comes from an OxTS GPS/INS unit. The KITTI
devkit turns its latitude and longitude into positions with
`latlonToMercator`: a spherical Mercator projection with the WGS84
equatorial radius, scaled by the cosine of the start latitude, applied to
WGS84 geodetic coordinates. On the ellipsoid that projection is not
conformal. At the latitude of Karlsruhe (49° N) it stretches north–south
distances by 0.099 % and shrinks east–west distances by 0.191 % (from the
ellipsoid's meridian and prime-vertical radii of curvature). The
ground-truth positions carry this direction-dependent scale error; the
orientations come straight from the inertial unit and do not.

A trajectory error of 0.5 % is the scale of the KITTI relative error
`t_rel`, so this note measures how much of the reported numbers comes from
the reference rather than from the trajectory. It covers 10 of the 11
sequences: the raw data of sequence 03 (drive 2011_09_26_0067) is not
published, and the OxTS log is needed.

## Checking that the ground truth is this projection

`scripts/download_kitti_oxts.py` fetches the OxTS logs of the raw drives
(only the OxTS part of each archive). After a rigid fit, the ground-truth
positions match the OxTS positions in the devkit projection to 0.5–6.0 cm
(RMS, horizontal) on sequences 00, 05, 06, 07, 08, 09, and to 11–21 cm on
01, 02, 04, 10. In UTM, a conformal projection, the same fit leaves 0.29 m
on sequence 00 and 0.21 m on 05
([map-quality.md](map-quality.md#5-georeferencing)). The larger residuals on
01, 02, 04, 10 were not investigated; they do not affect the correction
below, because the correction only needs to know where the trajectory lies
on the projection, and its gradient is the projection's anisotropy (0.3 %):
locating the trajectory 0.2 m off changes the correction by less than a
millimetre.

## A metric version of the ground truth

`src/benchmarks/metric_reference.py` maps each ground-truth position back
onto the devkit's Mercator plane (with the rigid fit above), inverts the
projection to latitude and longitude, and projects those with a conformal
projection of unit scale centred at the start, where distances are true to
better than 1e-7 within a few kilometres. A rigid fit brings the result back
into the ground-truth frame. Heights and orientations are kept, so the
metric version differs from the original only by the projection. The
correction moves the positions by 0.11–0.61 m RMS per sequence (up to 1.21
m).

## Effect on the metrics

`python -m scripts.eval_kitti_odometry references` evaluates the same
trajectories against both references.

| Seq | Correction RMS / max [m] | `t_rel` of the ground truth itself [%] | Stage 3 `t_rel` [%], KITTI → metric | Stage 3 SE(3) APE [m] | Stage 3 first-frame APE [m] |
|---|---:|---:|---:|---:|---:|
| 00 | 0.28 / 0.55 | 0.083 | 0.574 → 0.563 | 0.89 → 0.79 | 6.03 → 6.02 |
| 01 | 0.61 / 1.21 | 0.093 | 0.784 → 0.780 | 3.12 → 2.97 | 8.39 → 8.34 |
| 02 | 0.40 / 0.84 | 0.096 | 0.571 → 0.551 | 1.73 → 1.65 | 9.73 → 9.71 |
| 04 | 0.11 / 0.20 | 0.098 | 0.388 → 0.349 | 0.25 → 0.18 | 0.65 → 0.63 |
| 05 | 0.24 / 0.50 | 0.087 | 0.323 → 0.295 | 0.35 → 0.21 | 0.79 → 0.74 |
| 06 | 0.26 / 0.49 | 0.112 | 0.331 → 0.300 | 0.34 → 0.13 | 1.34 → 1.31 |
| 07 | 0.14 / 0.23 | 0.095 | 0.400 → 0.383 | 0.33 → 0.27 | 0.61 → 0.58 |
| 08 | 0.27 / 0.54 | 0.092 | 0.843 → 0.847 | 2.19 → 2.10 | 15.08 → 15.07 |
| 09 | 0.27 / 0.42 | 0.086 | 0.596 → 0.588 | 1.13 → 1.11 | 2.25 → 2.25 |
| 10 | 0.21 / 0.35 | 0.091 | 0.527 → 0.523 | 0.70 → 0.67 | 2.95 → 2.95 |

Pooled over the 10 sequences:

| Trajectory | `t_rel` against KITTI ground truth [%] | against the metric version [%] |
|---|---:|---:|
| Stage 2 (odometry) | 0.561 | 0.550 |
| Stage 3 (pose graph) | 0.590 | 0.578 |
| Official KISS-ICP, same machine | 0.562 | 0.551 |
| KITTI ground truth | – | 0.091 |

- **`t_rel` changes little.** Against the metric reference every trajectory
  scores 0.011–0.013 percentage points better. The ground truth's own error,
  0.091 % against its metric version, does not add to the trajectory's:
  errors in different directions combine like vectors, and most of it is
  lost in the trajectory's larger error. `r_rel` does not change, since the
  projection does not touch orientations.
- **Comparisons between trajectories are unaffected.** The gap between Stage
  2 and the official KISS-ICP changes by 0.0002 percentage points, so the
  check of the odometry wrapper in [evaluation.md](evaluation.md) stands.
- **Absolute errors of the best trajectories are dominated by it.** Where
  the trajectory is good, the reference's distortion is a large part of what
  SE(3) APE measures: on sequences 06, 05, 04 the metric reference lowers
  the Stage 3 SE(3) APE by 62 %, 40 %, 30 %. Where the trajectory error is
  metres (01, 02, 08), the change is within 10 %. First-frame APE barely
  moves: it is dominated by the start orientation
  ([evaluation.md](evaluation.md)).

## Consequences for this repository

- Trajectory numbers stay against the KITTI ground truth, as everywhere in
  the literature; this note gives the size of the reference error next to
  them.
- An SE(3) APE of a few decimetres against the KITTI ground truth is not a
  measurement of the trajectory alone. Differences below about the
  correction RMS of a sequence (0.1–0.6 m) are not resolved by this
  reference.
- The map is georeferenced with the OxTS positions projected to UTM, not
  with the ground truth ([map-quality.md](map-quality.md#5-georeferencing)).

## Reproduce

```bash
python scripts/download_kitti_oxts.py          # OxTS logs of 00-10 (03 is not published)
python -m scripts.eval_kitti_odometry run --variant kiss_icp_reference
python -m scripts.eval_kitti_odometry references  # writes benchmarks/kitti_references.csv
```
