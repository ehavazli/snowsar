#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures as cf
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional
import zipfile


DATE_PATTERNS = [
    # ...-YYMMDD.zip
    re.compile(r"-(\d{6})\.zip$", re.IGNORECASE),
    # ...-YYMMDD__D.zip (or any suffix before __D.zip)
    re.compile(r"-(\d{6})__D\.zip$", re.IGNORECASE),
    # Fallback: anywhere a 6-digit token right before .zip
    re.compile(r"(\d{6})\.zip$", re.IGNORECASE),
]


def ensure_dir(p: Path) -> bool:
    """Create directory if missing. Return True if created, False if existed."""
    try:
        p.mkdir(parents=True, exist_ok=False)
        return True
    except FileExistsError:
        return False


def extract_date_from_filename(filename: str) -> Optional[str]:
    """
    Extract a YYMMDD date string from filename using robust regex patterns.
    Returns 'YYMMDD' or None if not found/invalid.
    """
    for pat in DATE_PATTERNS:
        m = pat.search(filename)
        if m:
            date_str = m.group(1)
            # Validate it's a real date (YYMMDD)
            try:
                dt = datetime.strptime(date_str, "%y%m%d")
                return dt.strftime("%y%m%d")
            except ValueError:
                return None
    return None


def _safe_extractall(zf: zipfile.ZipFile, target: Path) -> None:
    """
    Securely extract all members of a ZipFile to 'target', preventing path traversal.
    """
    target_abs = target.resolve()
    for member in zf.infolist():
        # Normalize path separators and skip directory entries implicitly
        member_path = Path(member.filename)
        # Ignore absolute paths, drive letters, or parent traversal
        dest = (target_abs / member_path).resolve()
        try:
            dest.relative_to(target_abs)
        except ValueError:
            raise RuntimeError(f"Blocked path traversal: {member.filename}")
        if member.is_dir():
            dest.mkdir(parents=True, exist_ok=True)
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        with zf.open(member, "r") as src, open(dest, "wb") as dst:
            # chunked copy
            for chunk in iter(lambda: src.read(1024 * 1024), b""):
                dst.write(chunk)


def unpack_one(zip_path: Path, dates_dir: Path, overwrite: bool = False, logger: logging.Logger | None = None) -> str:
    """
    Unpack a single zip to dates/<YYMMDD>. Skips if folder exists and overwrite=False.
    Returns a short status string.
    """
    log = logger or logging.getLogger(__name__)
    date_token = extract_date_from_filename(zip_path.name)
    if not date_token:
        return f"SKIP (no-date): {zip_path.name}"

    out_dir = dates_dir / date_token
    if out_dir.exists() and not overwrite:
        return f"SKIP (exists): {out_dir.name}"

    ensure_dir(out_dir)

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            _safe_extractall(zf, out_dir)
        return f"DONE: {zip_path.name} -> {out_dir.name}"
    except Exception as e:
        # Remove empty dir on failure to avoid clutter
        try:
            if out_dir.exists() and not any(out_dir.iterdir()):
                out_dir.rmdir()
        except Exception:
            pass
        return f"FAIL: {zip_path.name} ({e})"


def iter_zip_files(folder: Path) -> Iterable[Path]:
    return (p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".zip")


def unpack_alos2(top_folder: str | Path, max_workers: int = os.cpu_count() or 4, overwrite: bool = False) -> None:
    """
    Unpack all ALOS-2 zips under <top_folder>/data/ into <top_folder>/dates/<YYMMDD>.
    Parallelized and safe by default.
    """
    top = Path(top_folder)
    data_dir = (top / "data").resolve()
    dates_dir = (top / "dates").resolve()
    ensure_dir(data_dir)
    ensure_dir(dates_dir)

    logging.info("Scanning for zip files in %s", data_dir)
    zips = list(iter_zip_files(data_dir))
    if not zips:
        logging.info("No .zip files found. Nothing to do.")
        return

    logging.info("Found %d zip files. Starting extraction with %d workers...", len(zips), max_workers)

    with cf.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(unpack_one, zp, dates_dir, overwrite, logging.getLogger(__name__)) for zp in zips]
        for fut in cf.as_completed(futures):
            logging.info(fut.result())

    logging.info("All done.")


def main():
    parser = argparse.ArgumentParser(description="Unpack ALOS-2 zip frames into date-based folders.")
    parser.add_argument("top_folder", help="Top folder containing 'data/' and 'dates/' (dates will be created if missing).")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 4, help="Number of parallel workers (default: CPU count).")
    parser.add_argument("--overwrite", action="store_true", help="Re-extract even if target date folder exists.")
    parser.add_argument("--loglevel", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.loglevel), format="%(levelname)s: %(message)s")
    unpack_alos2(args.top_folder, max_workers=args.workers, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
