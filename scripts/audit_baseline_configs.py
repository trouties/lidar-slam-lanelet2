#!/usr/bin/env python3
"""SUP-01 Stage B.5: read-only audit of all 3 baseline configs + launch files.

Catches the class of bugs that would make LIO-SAM / FAST-LIO2 / hdl_graph_slam
disagree with each other on the shared cached bag. Runs on host, no Docker
required. Exit 0 = no FAIL; 1 = at least one FAIL; warnings do not fail.

Usage::

    python -m scripts.audit_baseline_configs           # audit all 3
    python -m scripts.audit_baseline_configs --system fast_lio2
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASELINES = ROOT / "external" / "baselines"

EXPECTED_LIDAR_TOPIC = "/velodyne_points"
EXPECTED_IMU_TOPIC = "/imu/data"
EXPECTED_LIDAR_FRAME = "velodyne"


class Report:
    """Accumulator for audit findings. FAIL = hard stop; WARN = inform only."""

    def __init__(self, system: str) -> None:
        self.system = system
        self.lines: list[tuple[str, str, str]] = []

    def pass_(self, check: str, msg: str) -> None:
        self.lines.append(("PASS", check, msg))

    def warn(self, check: str, msg: str) -> None:
        self.lines.append(("WARN", check, msg))

    def fail(self, check: str, msg: str) -> None:
        self.lines.append(("FAIL", check, msg))

    def print(self) -> bool:
        print(f"\n=== {self.system} ===")
        for level, check, msg in self.lines:
            mark = {"PASS": "✓", "WARN": "!", "FAIL": "✗"}[level]
            print(f"  [{level}] {mark} {check}: {msg}")
        return not any(level == "FAIL" for level, _, _ in self.lines)


def _extract_yaml_list(text: str, key: str) -> list[float] | None:
    """Extract a flat list of floats from a ``key: [a, b, c, ...]`` YAML line.

    Tolerant of multi-line lists (comma-separated across newlines) and
    whitespace. Returns None if the key is not found or parses fail.
    """
    m = re.search(rf"{re.escape(key)}\s*:\s*\[([^\]]+)\]", text, re.DOTALL)
    if not m:
        return None
    try:
        return [float(x.strip()) for x in m.group(1).split(",") if x.strip()]
    except ValueError:
        return None


def _extract_yaml_scalar(text: str, key: str) -> str | None:
    m = re.search(rf"^\s*{re.escape(key)}\s*:\s*(.+?)\s*(?:#.*)?$", text, re.MULTILINE)
    return m.group(1).strip().strip('"').strip("'") if m else None


def _is_valid_rotation(R_flat: list[float], tol_det: float = 0.01, tol_trace: float = 0.1) -> tuple[bool, float, float]:
    """Return (ok, det, trace) — same thresholds as inject_kitti_extrinsic.py."""
    R = np.array(R_flat).reshape(3, 3)
    det = float(np.linalg.det(R))
    trace = float(np.trace(R))
    ok = (1.0 - tol_det < det < 1.0 + tol_det) and (abs(trace - 3.0) < tol_trace)
    return ok, det, trace


def audit_lio_sam() -> Report:
    r = Report("lio_sam")
    params_path = BASELINES / "lio_sam" / "config" / "params.yaml"
    tpl_path = BASELINES / "lio_sam" / "config" / "params.yaml.tpl"
    launch_path = BASELINES / "lio_sam" / "launch" / "kitti.launch"

    if not params_path.exists():
        r.fail("params.yaml", f"missing: {params_path}")
        return r
    text = params_path.read_text()

    # Topics
    lidar_topic = _extract_yaml_scalar(text, "pointCloudTopic")
    imu_topic = _extract_yaml_scalar(text, "imuTopic")
    lidar_frame = _extract_yaml_scalar(text, "lidarFrame")
    for key, got, want in [
        ("pointCloudTopic", lidar_topic, EXPECTED_LIDAR_TOPIC),
        ("imuTopic", imu_topic, EXPECTED_IMU_TOPIC),
        ("lidarFrame", lidar_frame, EXPECTED_LIDAR_FRAME),
    ]:
        if got == want:
            r.pass_(key, got)
        else:
            r.fail(key, f"got {got!r}, expected {want!r}")

    # Bug B regression guard
    heading_init = _extract_yaml_scalar(text, "useImuHeadingInitialization")
    if heading_init and heading_init.lower() == "false":
        r.pass_("useImuHeadingInitialization", "false (SUP-01 Bug B correct)")
    else:
        r.fail(
            "useImuHeadingInitialization",
            f"got {heading_init!r}, must be 'false' (Bug B: OxTS ENU yaw injection breaks GT alignment)",
        )

    # Extrinsic rotation validity + Rot == RPY invariant (post-inject)
    rot = _extract_yaml_list(text, "extrinsicRot")
    rpy = _extract_yaml_list(text, "extrinsicRPY")
    trans = _extract_yaml_list(text, "extrinsicTrans")
    tpl_has_placeholder = tpl_path.exists() and "${KITTI_EXT_ROT}" in tpl_path.read_text()
    if tpl_has_placeholder:
        r.pass_("extrinsic mode", "runtime envsubst (Stage C) — placeholders in params.yaml.tpl")
    elif rot and len(rot) == 9:
        ok, det, tr = _is_valid_rotation(rot)
        if ok:
            r.pass_("extrinsicRot validity", f"det={det:.4f} trace={tr:.4f}")
        else:
            r.fail("extrinsicRot validity", f"det={det:.4f} trace={tr:.4f} — NOT a near-identity rotation")
        if rpy and rot == rpy:
            r.pass_("extrinsicRPY == extrinsicRot", "byte-for-byte equal")
        else:
            r.fail(
                "extrinsicRPY == extrinsicRot",
                "DIVERGED — LIO-SAM uses RPY as exact, mismatch regresses APE by ~15 m",
            )
        if trans and len(trans) == 3:
            norm = float(np.linalg.norm(trans))
            if 0.3 < norm < 2.5:
                r.pass_("extrinsicTrans magnitude", f"|T|={norm:.3f} m (plausible)")
            else:
                r.warn("extrinsicTrans magnitude", f"|T|={norm:.3f} m (outside 0.3-2.5 m)")
    else:
        r.fail("extrinsicRot", "missing or not 9 floats")

    # Launch file existence
    if launch_path.exists():
        lt = launch_path.read_text()
        if "rosparam" in lt and "params.yaml" in lt:
            r.pass_("launch loads params.yaml", "rosparam load directive present")
        else:
            r.warn("launch loads params.yaml", "could not find rosparam load directive")
    else:
        r.fail("launch file", f"missing: {launch_path}")

    return r


def audit_fast_lio2() -> tuple[Report, list[float] | None, list[float] | None]:
    r = Report("fast_lio2")
    cfg_path = BASELINES / "fast_lio2" / "config" / "kitti.yaml"
    launch_path = BASELINES / "fast_lio2" / "launch" / "kitti.launch"

    if not cfg_path.exists():
        r.fail("config/kitti.yaml", f"missing: {cfg_path}")
        return r, None, None
    text = cfg_path.read_text()

    lidar_topic = _extract_yaml_scalar(text, "lid_topic")
    imu_topic = _extract_yaml_scalar(text, "imu_topic")
    for key, got, want in [
        ("lid_topic", lidar_topic, EXPECTED_LIDAR_TOPIC),
        ("imu_topic", imu_topic, EXPECTED_IMU_TOPIC),
    ]:
        if got == want:
            r.pass_(key, got)
        else:
            r.fail(key, f"got {got!r}, expected {want!r}")

    ext_R = _extract_yaml_list(text, "extrinsic_R")
    ext_T = _extract_yaml_list(text, "extrinsic_T")
    if ext_R and len(ext_R) == 9:
        ok, det, tr = _is_valid_rotation(ext_R)
        if ok:
            r.pass_("extrinsic_R validity", f"det={det:.4f} trace={tr:.4f}")
        else:
            r.fail("extrinsic_R validity", f"det={det:.4f} trace={tr:.4f}")
    else:
        r.fail("extrinsic_R", "missing or not 9 floats")

    if ext_T and len(ext_T) == 3:
        norm = float(np.linalg.norm(ext_T))
        if 0.3 < norm < 2.5:
            r.pass_("extrinsic_T magnitude", f"|T|={norm:.3f} m (plausible)")
        elif norm < 0.01:
            r.fail(
                "extrinsic_T magnitude",
                "|T|=0 m — KITTI IMU sits ~0.8 m offset from Velo; zero T = drift (SUP-01)",
            )
        else:
            r.warn("extrinsic_T magnitude", f"|T|={norm:.3f} m (outside 0.3-2.5 m)")
    else:
        r.fail("extrinsic_T", "missing or not 3 floats")

    # Launch file: the critical "extrinsic_T override" bug that nullifies 0.8 m offset
    if not launch_path.exists():
        r.fail("launch file", f"missing: {launch_path}")
    else:
        lt = launch_path.read_text()
        override_T = re.search(
            r'<param\s+name="mapping/extrinsic_T"\s+type="yaml"\s+value="\[([^\]]+)\]"',
            lt,
        )
        if override_T:
            vals = [float(x.strip()) for x in override_T.group(1).split(",")]
            if np.allclose(vals, [0.0, 0.0, 0.0]):
                r.fail(
                    "launch extrinsic_T override",
                    "kitti.launch sets extrinsic_T to [0,0,0], nullifying config. "
                    "This is the likely SUP-01 77 m drift cause on Seq 00. Remove the override.",
                )
            elif ext_T and np.allclose(vals, ext_T, atol=1e-3):
                r.pass_("launch extrinsic_T override", "override matches config values")
            else:
                r.warn(
                    "launch extrinsic_T override",
                    f"launch sets T={vals} while config has T={ext_T} — mismatch",
                )
        else:
            r.pass_("launch extrinsic_T override", "no launch-level override (config T is authoritative)")

    return r, ext_R, ext_T


def audit_hdl_graph_slam() -> Report:
    r = Report("hdl_graph_slam")
    cfg_path = BASELINES / "hdl_graph_slam" / "config" / "kitti.yaml"
    launch_path = BASELINES / "hdl_graph_slam" / "launch" / "kitti.launch"

    if not cfg_path.exists():
        r.fail("config/kitti.yaml", f"missing: {cfg_path}")
        return r
    text = cfg_path.read_text()

    for key, want_str in [
        ("enable_imu", "false"),
        ("enable_gps", "false"),
        ("enable_floor_detection", "false"),
        ("enable_loop_closure", "true"),
    ]:
        got = _extract_yaml_scalar(text, key)
        if got and got.lower() == want_str:
            r.pass_(key, want_str)
        else:
            r.fail(key, f"got {got!r}, expected {want_str!r}")

    # Registration sanity
    reg = _extract_yaml_scalar(text, "registration_method")
    if reg:
        r.pass_("registration_method", reg)

    if not launch_path.exists():
        r.fail("launch file", f"missing: {launch_path}")
    else:
        lt = launch_path.read_text()
        # Check topic remap to /velodyne_points
        if re.search(r'<remap\s+from="/velodyne_points"', lt) or "/velodyne_points" in lt:
            r.pass_("launch velodyne_points wiring", "topic referenced in launch")
        else:
            r.warn("launch velodyne_points wiring", "no /velodyne_points remap found")

    return r


def audit_cross_system(liosam_rot: list[float] | None, liosam_trans: list[float] | None,
                      fastlio_R: list[float] | None, fastlio_T: list[float] | None) -> Report:
    """FAST-LIO2 and LIO-SAM extrinsics must be transposes (the conventions are opposite)."""
    r = Report("cross-system extrinsic consistency (LIO-SAM vs FAST-LIO2)")
    if not (liosam_rot and fastlio_R and len(liosam_rot) == 9 and len(fastlio_R) == 9):
        r.warn(
            "R transpose relationship",
            "skipped: one of the rotation matrices unavailable",
        )
        return r
    R_lio = np.array(liosam_rot).reshape(3, 3)
    R_fast = np.array(fastlio_R).reshape(3, 3)
    # LIO-SAM: p_velo = R_lio @ p_imu + T_lio (R_lio = R_velo_imu)
    # FAST-LIO2: p_imu = R_fast @ p_velo + T_fast (R_fast = R_imu_velo = R_lio.T)
    R_fast_expected = R_lio.T
    err = float(np.abs(R_fast - R_fast_expected).max())
    if err < 1e-3:
        r.pass_("R transpose relationship", f"FAST-LIO2 R ≈ LIO-SAM Rᵀ (max diff {err:.2e})")
    else:
        r.fail(
            "R transpose relationship",
            f"FAST-LIO2 R differs from LIO-SAM Rᵀ by max {err:.4f} — one of them has wrong convention",
        )

    if liosam_trans and fastlio_T and len(liosam_trans) == 3 and len(fastlio_T) == 3:
        T_lio = np.array(liosam_trans)
        T_fast = np.array(fastlio_T)
        T_fast_expected = -R_fast_expected @ T_lio
        err_T = float(np.abs(T_fast - T_fast_expected).max())
        if err_T < 1e-2:
            r.pass_("T sign/rotation relationship", f"FAST-LIO2 T ≈ -Rᵀ·T_lio (max diff {err_T:.4f})")
        else:
            r.fail(
                "T sign/rotation relationship",
                f"FAST-LIO2 T={T_fast.tolist()} vs expected {T_fast_expected.tolist()} (max diff {err_T:.4f})",
            )
    return r


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--system",
        choices=["lio_sam", "fast_lio2", "hdl_graph_slam", "all"],
        default="all",
    )
    args = ap.parse_args()

    reports: list[Report] = []
    liosam_rot = liosam_trans = fastlio_R = fastlio_T = None

    if args.system in ("lio_sam", "all"):
        rep = audit_lio_sam()
        reports.append(rep)
        params_text = (BASELINES / "lio_sam" / "config" / "params.yaml").read_text()
        liosam_rot = _extract_yaml_list(params_text, "extrinsicRot")
        liosam_trans = _extract_yaml_list(params_text, "extrinsicTrans")

    if args.system in ("fast_lio2", "all"):
        rep, fastlio_R, fastlio_T = audit_fast_lio2()
        reports.append(rep)

    if args.system in ("hdl_graph_slam", "all"):
        reports.append(audit_hdl_graph_slam())

    if args.system == "all":
        reports.append(audit_cross_system(liosam_rot, liosam_trans, fastlio_R, fastlio_T))

    all_ok = True
    for rep in reports:
        all_ok &= rep.print()

    print()
    print("OVERALL: " + ("PASS" if all_ok else "FAIL"))
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
