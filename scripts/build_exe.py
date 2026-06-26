"""
build_exe.py — Build standalone executable via PyInstaller.

Usage:
    python scripts/build_exe.py
    python scripts/build_exe.py --onefile   # single .exe (slower startup)
    python scripts/build_exe.py --debug     # include console window
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent


def build(onefile: bool = False, debug: bool = False) -> None:
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--name", "HairPainter",
        "--icon", str(ROOT / "hairpainter" / "gui" / "icon.ico")
        if (ROOT / "hairpainter" / "gui" / "icon.ico").exists()
        else "NONE",
        "--add-data", f"{ROOT / 'hairpainter' / 'models'}{';' if sys.platform == 'win32' else ':'}hairpainter/models",
        "--hidden-import", "easyocr",
        "--hidden-import", "skimage.filters.frangi",
        "--hidden-import", "skimage.morphology",
        "--hidden-import", "scipy.ndimage",
        "--hidden-import", "tifffile",
        "--hidden-import", "PyQt6",
    ]

    if onefile:
        cmd.append("--onefile")
    else:
        cmd.append("--onedir")

    if not debug:
        cmd.append("--noconsole")

    cmd.append(str(ROOT / "hairpainter" / "__main__.py"))

    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd, cwd=str(ROOT))
    if result.returncode != 0:
        print("Build FAILED.")
        sys.exit(1)

    platform = "windows" if sys.platform == "win32" else "linux"
    exe_name = "HairPainter.exe" if sys.platform == "win32" else "HairPainter"
    dist = ROOT / "dist"

    if onefile:
        exe = dist / exe_name
    else:
        exe = dist / "HairPainter" / exe_name

    print(f"\nBuild succeeded: {exe}")
    print(f"Executable size: {exe.stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--onefile", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    build(onefile=args.onefile, debug=args.debug)
