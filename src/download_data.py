"""
download_data.py — Download CWRU .mat files into data/raw/

Uses the URLs from the multivariate-cwru package (Zenodo mirrors).
Downloads 12kHz Drive End fault files at 1797 RPM + all Normal baselines.
Files are saved with descriptive names compatible with load_data.py.
"""

import os
import urllib.request
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

# (descriptive_filename, zenodo_url)
FILES = [
    # Normal baseline (all four motor loads)
    ("normal_1797.mat", "https://zenodo.org/records/10986655/files/97.mat?download=1"),
    ("normal_1772.mat", "https://zenodo.org/records/10986655/files/98.mat?download=1"),
    ("normal_1750.mat", "https://zenodo.org/records/10986655/files/99.mat?download=1"),
    ("normal_1730.mat", "https://zenodo.org/records/10986655/files/100.mat?download=1"),
    # Ball fault — 12kHz Drive End, 1797 RPM
    ("B007_1797.mat",   "https://zenodo.org/records/10986655/files/118.mat?download=1"),
    ("B014_1797.mat",   "https://zenodo.org/records/10986655/files/185.mat?download=1"),
    ("B021_1797.mat",   "https://zenodo.org/records/10986655/files/222.mat?download=1"),
    # Inner race fault — 12kHz Drive End, 1797 RPM
    ("IR007_1797.mat",  "https://zenodo.org/records/10986655/files/105.mat?download=1"),
    ("IR014_1797.mat",  "https://zenodo.org/records/10986655/files/169.mat?download=1"),
    ("IR021_1797.mat",  "https://zenodo.org/records/10986655/files/209.mat?download=1"),
    # Outer race fault (6 o'clock) — 12kHz Drive End, 1797 RPM
    ("OR007_1797.mat",  "https://zenodo.org/records/10986655/files/130.mat?download=1"),
    ("OR014_1797.mat",  "https://zenodo.org/records/10986655/files/197.mat?download=1"),
    ("OR021_1797.mat",  "https://zenodo.org/records/10986655/files/234.mat?download=1"),
]


def download_all(out_dir: Path = RAW_DIR, force: bool = False) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    total = len(FILES)

    for i, (fname, url) in enumerate(FILES, 1):
        dest = out_dir / fname
        if dest.exists() and not force:
            size_kb = dest.stat().st_size // 1024
            print(f"[{i:02d}/{total}] SKIP  {fname}  (already exists, {size_kb} KB)")
            continue

        print(f"[{i:02d}/{total}] GET   {fname} ...", end=" ", flush=True)
        try:
            urllib.request.urlretrieve(url, dest)
            size_kb = dest.stat().st_size // 1024
            print(f"done  ({size_kb} KB)")
        except Exception as exc:
            print(f"FAILED — {exc}")
            if dest.exists():
                dest.unlink()  # remove partial download


if __name__ == "__main__":
    print(f"Downloading CWRU dataset to: {RAW_DIR}\n")
    download_all()
    print("\nDone.")
