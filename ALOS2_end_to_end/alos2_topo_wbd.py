#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import os
import shlex
import subprocess
from dataclasses import dataclass
from math import ceil, floor
from pathlib import Path
from typing import Optional, Tuple, List


# ---------- Data types ----------

@dataclass(frozen=True)
class BBox:
    """Bounding box in degrees: (min_lat, max_lat, min_lon, max_lon)."""
    min_lat: float
    max_lat: float
    min_lon: float
    max_lon: float

    @property
    def flo_ceil(self) -> Tuple[int, int, int, int]:
        """Return (MINLAT_LO, MAXLAT_HI, MINLON_LO, MAXLON_HI) as integers for ISCE apps."""
        return (floor(self.min_lat), ceil(self.max_lat), floor(self.min_lon), ceil(self.max_lon))


# ---------- Utilities ----------

def run_cmd(cmd: List[str], cwd: Optional[Path] = None, logger: Optional[logging.Logger] = None) -> subprocess.CompletedProcess:
    """Run a shell command, capture stdout/stderr, raise on nonzero exit."""
    log = logger or logging.getLogger(__name__)
    log.debug("Running: %s", " ".join(shlex.quote(c) for c in cmd))
    cp = subprocess.run(cmd, cwd=str(cwd) if cwd else None, text=True, capture_output=True)
    if cp.returncode != 0:
        log.error("Command failed (%s): %s", cp.returncode, " ".join(cmd))
        log.error("STDERR:\n%s", cp.stderr.strip())
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")
    if cp.stdout:
        log.debug("STDOUT:\n%s", cp.stdout.strip())
    return cp


def _clip_latlon(lat: float, lon: float) -> Tuple[float, float]:
    """Clip lat to [-90, 90] and lon to [-180, 180]."""
    lat = max(-90.0, min(90.0, lat))
    lon = max(-180.0, min(180.0, lon))
    return lat, lon


def _expand_bbox(bbox: BBox, margin_deg: float) -> BBox:
    """Expand bbox by margin on each side, clipped to world limits."""
    min_lat, min_lon = _clip_latlon(bbox.min_lat - margin_deg, bbox.min_lon - margin_deg)
    max_lat, max_lon = _clip_latlon(bbox.max_lat + margin_deg, bbox.max_lon + margin_deg)
    return BBox(min_lat=min_lat, max_lat=max_lat, min_lon=min_lon, max_lon=max_lon)


def _find_token_path(text: str, needle: str) -> Optional[str]:
    """
    From dem.py / wbd.py stdout text, find a token that contains `needle`
    (e.g., '/wgs84/' or '/swbdLat/'), and return it.
    """
    for line in text.splitlines():
        if needle in line:
            tokens = line.strip().split()
            for tok in reversed(tokens):
                if needle in tok:
                    return tok
    return None


# ---------- Public API: bbox from user only ----------

def get_user_input_bbox(ctx: dict, margin_deg: float = 0.05) -> Tuple[BBox, BBox]:
    """
    Read bbox from ctx (must include min_lat,max_lat,min_lon,max_lon),
    return (raw_bbox, expanded_bbox).
    """
    try:
        min_lat = float(ctx["min_lat"])
        max_lat = float(ctx["max_lat"])
        min_lon = float(ctx["min_lon"])
        max_lon = float(ctx["max_lon"])
    except Exception:
        raise ValueError("ctx must contain numeric 'min_lat','max_lat','min_lon','max_lon' values.")

    raw = BBox(min_lat=min_lat, max_lat=max_lat, min_lon=min_lon, max_lon=max_lon)
    return raw, _expand_bbox(raw, margin_deg)


def get_bbox(ctx: dict, margin_deg: float = 0.05, logger: Optional[logging.Logger] = None) -> Tuple[BBox, BBox]:
    """
    Strict: require user bbox. No auto-computation from products.
    Returns (raw_bbox, expanded_bbox).
    """
    _ = logger  # reserved for parity with earlier signature
    return get_user_input_bbox(ctx, margin_deg=margin_deg)


# ---------- Public API: downloads ----------

def _require_isce_home(isce_home: Optional[Path]) -> Path:
    raw = (
        str(isce_home).strip()
        if isce_home is not None
        else os.environ.get("ISCE_HOME", "").strip()
    )
    if not raw:
        raise EnvironmentError(
            "ISCE_HOME is not set. Set env var or pass isce_home=..."
        )
    p = Path(raw).expanduser().resolve()
    if not p.exists():
        raise EnvironmentError(f"ISCE_HOME does not exist: {p}")

    apps_dir = p / "applications"
    required_apps = ("dem.py", "wbd.py", "fixImageXml.py")
    missing = [name for name in required_apps if not (apps_dir / name).exists()]
    if missing:
        raise EnvironmentError(
            f"ISCE_HOME missing required applications under {apps_dir}: {missing}"
        )
    return p


def download_dem(work_dir: Path, bbox: BBox, isce_home: Optional[Path] = None, arcsec: int = 1,
                 logger: Optional[logging.Logger] = None) -> Path:
    log = logger or logging.getLogger(__name__)
    isce_home = _require_isce_home(isce_home)

    subdir = "topo" if arcsec == 1 else "topo/3arcsec"
    out_dir = (work_dir / subdir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    MINLAT_LO, MAXLAT_HI, MINLON_LO, MAXLON_HI = bbox.flo_ceil
    cmd = [
        str(isce_home / "applications" / "dem.py"),
        "-a", "stitch",
        "-b", str(MINLAT_LO), str(MAXLAT_HI), str(MINLON_LO), str(MAXLON_HI),  # <-- split args!
        "-r",
        "-s", str(arcsec),
        "-f",
        "-c",
    ]
    cp = run_cmd(cmd, cwd=out_dir, logger=log)
    (out_dir / ("dem3.txt" if arcsec == 3 else "dem.txt")).write_text(cp.stdout or "", encoding="utf-8")

    wgs84_token = _find_token_path(cp.stdout, ".wgs84")
    if wgs84_token:
        wgs84_path = (out_dir / Path(wgs84_token)).resolve() if not Path(wgs84_token).is_absolute() else Path(wgs84_token)
        log.info("WGS84 detected: %s", wgs84_path)
        if wgs84_path.exists():
            fix_cmd = [str(isce_home / "applications" / "fixImageXml.py"), "--full", "-i", str(wgs84_path)]
            run_cmd(fix_cmd, logger=log)
            return wgs84_path
        log.warning("WGS84 file not found on disk: %s", wgs84_path)
    else:
        log.warning("Could not detect WGS84 path from dem.py output.")
    return out_dir


def download_wbd(work_dir: Path, bbox: BBox, isce_home: Optional[Path] = None,
                 logger: Optional[logging.Logger] = None) -> Path:
    log = logger or logging.getLogger(__name__)
    isce_home = _require_isce_home(isce_home)

    out_dir = (work_dir / "topo").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    MINLAT_LO, MAXLAT_HI, MINLON_LO, MAXLON_HI = bbox.flo_ceil
    cmd = [
        str(isce_home / "applications" / "wbd.py"),
        str(MINLAT_LO), str(MAXLAT_HI), str(MINLON_LO), str(MAXLON_HI),  # <-- split args!
        "1",
    ]
    cp = run_cmd(cmd, cwd=out_dir, logger=log)
    (out_dir / "wbd.txt").write_text(cp.stdout or "", encoding="utf-8")

    swbd_token = _find_token_path(cp.stdout, "swbdLat")
    if swbd_token:
        swbd_path = (out_dir / Path(swbd_token)).resolve() if not Path(swbd_token).is_absolute() else Path(swbd_token)
        log.info("WBD detected: %s", swbd_path)
        if swbd_path.exists():
            fix_cmd = [str(isce_home / "applications" / "fixImageXml.py"), "--full", "-i", str(swbd_path)]
            run_cmd(fix_cmd, logger=log)
            return swbd_path
        log.warning("WBD file not found on disk: %s", swbd_path)
    else:
        log.warning("Could not detect WBD path from wbd.py output.")
    return out_dir


# ---------- High-level orchestration ----------

def prepare_topo(ctx: dict,
                 use_dem: bool = True,
                 use_dem3: bool = False,
                 use_wbd: bool = True,
                 bbox_override: Optional[Tuple[float, float, float, float]] = None,
                 margin_deg: float = 0.05,
                 isce_home: Optional[str] = None,
                 logger: Optional[logging.Logger] = None) -> dict:
    """
    High-level runner using **user-provided bbox only**.
    - If bbox_override is provided, it wins.
    - Else, reads bbox from ctx[min_lat,max_lat,min_lon,max_lon].
    - Expands bbox by margin_deg to ensure full coverage.
    - Downloads requested assets (DEM 1", DEM 3", WBD).
    Returns dict with bbox info and output paths.
    """
    log = logger or logging.getLogger(__name__)
    work_dir = Path(ctx["work_dir"]).expanduser().resolve()

    if bbox_override:
        raw = BBox(*bbox_override)
        raw, expanded = raw, _expand_bbox(raw, margin_deg)
    else:
        raw, expanded = get_bbox(ctx, margin_deg=margin_deg, logger=log)

    results = {
        "bbox_raw": raw,
        "bbox_expanded": expanded,
        "work_dir": str(work_dir),
        "outputs": {},
    }

    if use_dem:
        dem1_path = download_dem(work_dir, expanded, isce_home=Path(isce_home) if isce_home else None, arcsec=1, logger=log)
        results["outputs"]["dem_1arcsec"] = str(dem1_path)

    if use_dem3:
        dem3_path = download_dem(work_dir, expanded, isce_home=Path(isce_home) if isce_home else None, arcsec=3, logger=log)
        results["outputs"]["dem_3arcsec"] = str(dem3_path)

    if use_wbd:
        wbd_path = download_wbd(work_dir, expanded, isce_home=Path(isce_home) if isce_home else None, logger=log)
        results["outputs"]["wbd"] = str(wbd_path)

    return results


# ---------- CLI ----------

def _parse_bbox_arg(bbox_str: str) -> Tuple[float, float, float, float]:
    parts = [p.strip() for p in bbox_str.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("bbox must be 'min_lat,max_lat,min_lon,max_lon'")
    try:
        return tuple(float(p) for p in parts)  # type: ignore[return-value]
    except ValueError:
        raise argparse.ArgumentTypeError("bbox values must be floats")


def main():
    parser = argparse.ArgumentParser(description="Prepare topo inputs (DEM / WBD) using a user-provided bbox only.")
    parser.add_argument("--work-dir", required=True, help="Top working directory (e.g., path with dates/ etc.).")
    parser.add_argument("--isce-home", default=os.environ.get("ISCE_HOME", ""), help="Path to ISCE_HOME (defaults to env var).")
    parser.add_argument("--bbox", type=_parse_bbox_arg, required=True,
                        help="Required bbox 'min_lat,max_lat,min_lon,max_lon'.")
    parser.add_argument("--margin", type=float, default=0.05, help="Margin in degrees to expand the bbox (default 0.05).")
    parser.add_argument("--no-dem", action="store_true", help="Skip DEM (1 arcsec) download.")
    parser.add_argument("--dem3", action="store_true", help="Also download 3 arcsec DEM.")
    parser.add_argument("--no-wbd", action="store_true", help="Skip WBD download.")
    parser.add_argument("--loglevel", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.loglevel), format="%(levelname)s: %(message)s")

    ctx = {
        "work_dir": args.work_dir,
        "min_lat": args.bbox[0],
        "max_lat": args.bbox[1],
        "min_lon": args.bbox[2],
        "max_lon": args.bbox[3],
    }

    results = prepare_topo(
        ctx=ctx,
        use_dem=not args.no_dem,
        use_dem3=args.dem3,
        use_wbd=not args.no_wbd,
        bbox_override=None,   # CLI always uses ctx bbox
        margin_deg=args.margin,
        isce_home=args.isce_home or None,
    )

    raw = results["bbox_raw"]
    expd = results["bbox_expanded"]
    print(f"Raw bbox      : {raw}")
    print(f"Expanded bbox : {expd}")
    for k, v in results["outputs"].items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
