"""Package the first frames of a KITTI Odometry sequence for the Docker reproduction.

Run once by the maintainer; the tarball is published as a release asset and
downloaded by ``scripts/docker_entrypoint.sh``. Layout inside the tarball
(extract under the dataset root, ``tar -xzf kitti_seq00_200.tar.gz -C /data``):

    sequences/<SEQ>/velodyne/000000.bin ...
    sequences/<SEQ>/calib.txt
    sequences/<SEQ>/times.txt
    poses/<SEQ>.txt   (full-sequence ground truth)

Also writes ``<output>.sha256``.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import tarfile
from pathlib import Path


def sha256_of_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            buf = f.read(chunk_size)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def build_tarball(
    kitti_root: Path,
    sequence: str,
    num_frames: int,
    output: Path,
) -> None:
    seq_dir = kitti_root / "sequences" / sequence
    velo_dir = seq_dir / "velodyne"
    calib = seq_dir / "calib.txt"
    times = seq_dir / "times.txt"
    poses = kitti_root / "poses" / f"{sequence}.txt"

    for required in (velo_dir, calib, times, poses):
        if not required.exists():
            raise FileNotFoundError(f"required KITTI file missing: {required}")

    bins = sorted(velo_dir.glob("*.bin"))
    if len(bins) < num_frames:
        raise ValueError(
            f"sequence {sequence} has only {len(bins)} velodyne frames, need {num_frames}"
        )
    bins = bins[:num_frames]

    output.parent.mkdir(parents=True, exist_ok=True)

    print(f"writing {output} (seq={sequence}, n={num_frames})")
    with tarfile.open(output, "w:gz") as tar:
        for b in bins:
            arcname = f"sequences/{sequence}/velodyne/{b.name}"
            tar.add(b, arcname=arcname)
        tar.add(calib, arcname=f"sequences/{sequence}/calib.txt")
        tar.add(times, arcname=f"sequences/{sequence}/times.txt")
        tar.add(poses, arcname=f"poses/{sequence}.txt")

    size_mb = output.stat().st_size / (1024 * 1024)
    digest = sha256_of_file(output)
    print(f"size={size_mb:.1f} MB  sha256={digest}")

    sha_path = output.with_suffix(output.suffix + ".sha256")
    sha_path.write_text(f"{digest}  {output.name}\n")

    print(f"wrote {sha_path}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--kitti-root",
        type=Path,
        default=Path.home() / "data/kitti/odometry/dataset",
        help="KITTI Odometry dataset root (contains sequences/ and poses/)",
    )
    ap.add_argument("--sequence", default="00", help="KITTI sequence id (default: 00)")
    ap.add_argument(
        "--num-frames",
        type=int,
        default=200,
        help="number of leading velodyne frames to include (default: 200)",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=Path("/tmp/kitti_seq00_200.tar.gz"),
        help="output tarball path",
    )
    args = ap.parse_args(argv)

    build_tarball(
        kitti_root=args.kitti_root.expanduser(),
        sequence=args.sequence,
        num_frames=args.num_frames,
        output=args.output,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
