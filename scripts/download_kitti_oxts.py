#!/usr/bin/env python3
"""Download the OxTS (GNSS/INS) logs of the KITTI raw drives behind the odometry sequences.

The OxTS log of a drive is a few megabytes, but it ships inside the drive's
``*_extract.zip``, which holds every sensor and is several gigabytes. The
archive is read in place over HTTP range requests: ``zipfile`` gets a
seekable file object that fetches only the blocks it touches (the central
directory at the end, then the ``oxts/`` members), so only the OxTS part is
transferred.

Usage:
  python scripts/download_kitti_oxts.py --sequences 00 01 02
"""

from __future__ import annotations

import argparse
import io
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.kitti_raw import ODOM_TO_RAW  # noqa: E402

URL_TEMPLATE = (
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/"
    "{date}_drive_{drive}/{date}_drive_{drive}_extract.zip"
)
CALIB_URL_TEMPLATE = "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/{date}_calib.zip"


class HttpRangeFile(io.RawIOBase):
    """Read-only, seekable view of a remote file, fetched in cached blocks."""

    def __init__(self, url: str, block_size: int = 4 << 20, max_blocks: int = 16) -> None:
        self.url = url
        self.block_size = block_size
        self.max_blocks = max_blocks
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=60) as r:
            self.size = int(r.headers["Content-Length"])
        self.pos = 0
        self._blocks: dict[int, bytes] = {}
        self.bytes_fetched = 0

    def _block(self, k: int) -> bytes:
        if k not in self._blocks:
            lo = k * self.block_size
            hi = min(lo + self.block_size, self.size) - 1
            req = urllib.request.Request(self.url, headers={"Range": f"bytes={lo}-{hi}"})
            with urllib.request.urlopen(req, timeout=120) as r:
                data = r.read()
            self.bytes_fetched += len(data)
            if len(self._blocks) >= self.max_blocks:
                self._blocks.pop(next(iter(self._blocks)))
            self._blocks[k] = data
        return self._blocks[k]

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self.pos, io.SEEK_END: self.size}[whence]
        self.pos = base + offset
        return self.pos

    def read(self, n: int = -1) -> bytes:
        if n is None or n < 0:
            n = self.size - self.pos
        n = max(0, min(n, self.size - self.pos))
        out = bytearray()
        while len(out) < n:
            k, off = divmod(self.pos, self.block_size)
            chunk = self._block(k)[off : off + n - len(out)]
            out += chunk
            self.pos += len(chunk)
        return bytes(out)

    def readinto(self, b) -> int:
        data = self.read(len(b))
        b[: len(data)] = data
        return len(data)


def _extract(url: str, dest: Path, keep) -> tuple[int, int]:
    f = HttpRangeFile(url)
    with zipfile.ZipFile(f) as zf:
        members = [m for m in zf.infolist() if keep(m.filename)]
        members.sort(key=lambda m: m.header_offset)
        for m in members:
            zf.extract(m, dest)
    return len(members), f.bytes_fetched


def download_oxts(seq: str, raw_root: Path) -> bool:
    date, drive = ODOM_TO_RAW[seq]
    oxts_dir = raw_root / date / f"{date}_drive_{drive}_extract" / "oxts"
    if not (raw_root / date / "calib_imu_to_velo.txt").exists():
        n, nbytes = _extract(
            CALIB_URL_TEMPLATE.format(date=date), raw_root, lambda name: name.endswith(".txt")
        )
        print(f"  [{seq}] calibration of {date}: {n} files, {nbytes / 1e6:.1f} MB transferred")
    if oxts_dir.is_dir() and any((oxts_dir / "data").glob("*.txt")):
        print(f"  [{seq}] already present: {oxts_dir}")
        return True
    url = URL_TEMPLATE.format(date=date, drive=drive)
    try:
        # Member names start with the date directory, so extract into the raw root.
        n, nbytes = _extract(url, raw_root, lambda name: "/oxts/" in name)
    except urllib.error.HTTPError as e:
        # Not every drive was published: sequence 03 (2011_09_26_drive_0067) is missing.
        print(f"  [{seq}] {url.rsplit('/', 1)[1]}: HTTP {e.code}")
        return False
    name = url.rsplit("/", 1)[1]
    print(f"  [{seq}] {n} OxTS files from {name}, {nbytes / 1e6:.1f} MB transferred")
    return oxts_dir.is_dir()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sequences", nargs="+", default=sorted(ODOM_TO_RAW))
    parser.add_argument("--raw-root", default="~/data/kitti_raw")
    args = parser.parse_args()
    raw_root = Path(args.raw_root).expanduser()
    failed = [s for s in args.sequences if not download_oxts(s, raw_root)]
    if failed:
        raise SystemExit(f"failed: {failed}")


if __name__ == "__main__":
    main()
