#!/usr/bin/env python3
"""
Compute Local Incidence Angle (LIA) from ISCE radar-grid LOS + HGT + lon/lat,
using 3D surface normals in local ENU (no East/North slope assumption).

Inputs (all same raster shape):
  --los : 2-band ISCE .los
          band 1 = incidence angle (deg, measured FROM VERTICAL, always +)
          band 2 = azimuth angle  (deg, measured FROM NORTH, ANTI-CLOCKWISE)
  --hgt : ISCE .hgt (height in meters)
  --lon : longitude raster (degrees)
  --lat : latitude  raster (degrees)

Outputs (GeoTIFFs on the same grid):
  - localIncAngle.tif  : Local incidence angle in degrees
  - shadowMask.tif     : 1 where cos(LIA) < 0  (i.e., LIA > 90°, back-facing / shadow), else 0
  - layoverMask.tif    : 1 where directional slope toward sensor > incidence angle (rough heuristic)

IMPORTANT CONVENTIONS:
  * Azimuth is assumed CCW from North (ISCE). If your data were CW, the east component sign would flip.
  * LOS vector is constructed as ground→satellite in ENU.
  * Normals are computed from central differences of neighbor 3D positions in the
    local ENU frame (via ECEF→ENU at each pixel), so this works in radar grid.

ASSUMPTION MADE ALWAYS TRUE (per user request):
  We ALWAYS emit shadow and layover masks alongside LIA (no CLI flag to disable).
"""

import argparse

import numpy as np
import rasterio

# ---------- WGS84 ----------
WGS84_A = 6378137.0
WGS84_E2 = 6.69437999014e-3


def llh_to_ecef(lon_deg, lat_deg, h_m):
    lon = np.deg2rad(lon_deg)
    lat = np.deg2rad(lat_deg)
    sinl, cosl = np.sin(lat), np.cos(lat)
    sinL, cosL = np.sin(lon), np.cos(lon)
    N = WGS84_A / np.sqrt(1.0 - WGS84_E2 * sinl * sinl)
    X = (N + h_m) * cosl * cosL
    Y = (N + h_m) * cosl * sinL
    Z = (N * (1.0 - WGS84_E2) + h_m) * sinl
    return X, Y, Z


def ecef_to_enu(x, y, z, lon0_deg, lat0_deg, h0_m):
    """
    Vectorized ECEF -> ENU relative to per-pixel reference (lon0,lat0,h0).
    All inputs are arrays of the same shape.
    """
    x0, y0, z0 = llh_to_ecef(lon0_deg, lat0_deg, h0_m)
    dx = x - x0
    dy = y - y0
    dz = z - z0

    lon0 = np.deg2rad(lon0_deg)
    lat0 = np.deg2rad(lat0_deg)
    sl, cl = np.sin(lat0), np.cos(lat0)
    sL, cL = np.sin(lon0), np.cos(lon0)

    # ECEF -> ENU rotation applied component-wise
    e = -sL * dx + cL * dy
    n = -sl * cL * dx + (-sl * sL) * dy + cl * dz
    u = cl * cL * dx + cl * sL * dy + sl * dz
    return e, n, u


def los_enu_ground_to_sat_isce_ccw(inc_deg, az_deg):
    """
    ISCE convention:
      incidence from vertical (+), azimuth from North, ANTI-CLOCKWISE.
    Ground->satellite vector in local ENU:
    """
    th = np.deg2rad(inc_deg)
    ps = np.deg2rad(az_deg)
    sE = -np.sin(th) * np.sin(ps)  # note minus for CCW azimuth
    sN = np.sin(th) * np.cos(ps)
    sU = -np.cos(th)
    nrm = np.sqrt(sE * sE + sN * sN + sU * sU)
    nrm = np.where(nrm == 0, 1.0, nrm)
    return (
        (sE / nrm).astype(np.float32),
        (sN / nrm).astype(np.float32),
        (sU / nrm).astype(np.float32),
    )


def surface_normal_from_llh_central(lon, lat, h):
    """
    Build 3D surface normals via central differences in the radar grid,
    projecting neighbor ECEF positions into the current pixel's local ENU.
    Returns unit normal (nE,nN,nU) with nU>=0; NaN at 1-pixel border.
    """
    # Central neighbors
    lon_xp = np.roll(lon, -1, 1)
    lat_xp = np.roll(lat, -1, 1)
    h_xp = np.roll(h, -1, 1)
    lon_xm = np.roll(lon, 1, 1)
    lat_xm = np.roll(lat, 1, 1)
    h_xm = np.roll(h, 1, 1)
    lon_yp = np.roll(lon, -1, 0)
    lat_yp = np.roll(lat, -1, 0)
    h_yp = np.roll(h, -1, 0)
    lon_ym = np.roll(lon, 1, 0)
    lat_ym = np.roll(lat, 1, 0)
    h_ym = np.roll(h, 1, 0)

    # ECEF positions
    Xxp, Yxp, Zxp = llh_to_ecef(lon_xp, lat_xp, h_xp)
    Xxm, Yxm, Zxm = llh_to_ecef(lon_xm, lat_xm, h_xm)
    Xyp, Yyp, Zyp = llh_to_ecef(lon_yp, lat_yp, h_yp)
    Xym, Yym, Zym = llh_to_ecef(lon_ym, lat_ym, h_ym)

    # Project neighbors into **local ENU of the center pixel**
    ExP, NxP, UxP = ecef_to_enu(Xxp, Yxp, Zxp, lon, lat, h)
    ExM, NxM, UxM = ecef_to_enu(Xxm, Yxm, Zxm, lon, lat, h)
    EyP, NyP, UyP = ecef_to_enu(Xyp, Yyp, Zyp, lon, lat, h)
    EyM, NyM, UyM = ecef_to_enu(Xym, Yym, Zym, lon, lat, h)

    # Tangent vectors (central differences) in ENU
    txE = ExP - ExM
    txN = NxP - NxM
    txU = UxP - UxM
    tyE = EyP - EyM
    tyN = NyP - NyM
    tyU = UyP - UyM

    # Normal = cross(tx, ty); order matters for sign (right-handed ENU)
    nE = txN * tyU - txU * tyN
    nN = txU * tyE - txE * tyU
    nU = txE * tyN - txN * tyE

    # Normalize and enforce upward normal (nU>=0)
    nrm = np.sqrt(nE * nE + nN * nN + nU * nU)
    nrm = np.where(nrm == 0, np.nan, nrm)
    nE /= nrm
    nN /= nrm
    nU /= nrm
    flip = nU < 0
    nE[flip] *= -1
    nN[flip] *= -1
    nU[flip] *= -1

    # Invalidate 1-pixel border (rolled neighbors wrap)
    for A in (nE, nN, nU):
        A[:, 0] = A[:, -1] = np.nan
        A[0, :] = A[-1, :] = np.nan

    return nE.astype(np.float32), nN.astype(np.float32), nU.astype(np.float32)


def lia_and_costh_from_normal(nE, nN, nU, sE, sN, sU):
    """
    cos(theta_LIA) = n · (-s) = -(n·s)
    returns (lia_deg, costh)
    """
    costh = np.clip(-(nE * sE + nN * sN + nU * sU), -1.0, 1.0)
    lia = np.degrees(np.arccos(costh)).astype(np.float32)
    return lia, costh


def compute_layover_mask(nE, nN, nU, sE, sN, sU, lia_deg, inc_deg):
    """
    Rough layover heuristic:
      front-facing (LIA < 90°) AND terrain faces LOS more steeply than incidence.
    Implemented via normal tilt projected onto look azimuth and compared to tan(inc).
    """
    hnorm = np.sqrt(sE * sE + sN * sN)
    sEh = np.where(hnorm > 0, sE / hnorm, 0.0)
    sNh = np.where(hnorm > 0, sN / hnorm, 0.0)

    n_h_proj = nE * sEh + nN * sNh  # horizontal component of normal along LOS azimuth
    tilt_tan = np.where(nU > 0, np.abs(n_h_proj) / nU, np.inf)  # ~ tan(surface tilt)
    front = lia_deg < 90.0
    return (front & (tilt_tan > np.tan(np.deg2rad(inc_deg)))).astype(np.uint8)


def write_geotiff_like(path, reference_profile, array, dtype, nodata=None):
    prof = reference_profile.copy()
    prof.update(count=1, dtype=dtype, compress="deflate", nodata=nodata)
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(array.astype(dtype), 1)


def main():
    ap = argparse.ArgumentParser(
        description="LIA from ISCE radar-grid via 3D normals (ALWAYS writes shadow/layover masks)."
    )
    ap.add_argument(
        "--los",
        required=True,
        help=".los (2 bands: inc[from vertical], az[CCW from North])",
    )
    ap.add_argument("--hgt", required=True, help=".hgt (meters)")
    ap.add_argument("--lon", required=True, help="lon raster (deg)")
    ap.add_argument("--lat", required=True, help="lat raster (deg)")
    ap.add_argument("--out", default="localIncAngle.tif")
    ap.add_argument("--shadow-out", default="shadowMask.tif")
    ap.add_argument("--layover-out", default="layoverMask.tif")
    args = ap.parse_args()

    # Read inputs
    with rasterio.open(args.los) as ds_los:
        los = ds_los.read().astype(np.float32)
        if los.shape[0] < 2:
            raise ValueError("LOS must have 2 bands: [incidence, azimuth].")
        inc, az = los[0], los[1]
        H, W = inc.shape
        out_prof = ds_los.profile

    with rasterio.open(args.hgt) as ds_hgt:
        h = ds_hgt.read(1).astype(np.float32)
        if h.shape != (H, W):
            raise ValueError("HGT shape must match LOS shape.")

    with rasterio.open(args.lon) as ds_lon, rasterio.open(args.lat) as ds_lat:
        lon = ds_lon.read(1).astype(np.float64)
        lat = ds_lat.read(1).astype(np.float64)
        if lon.shape != (H, W) or lat.shape != (H, W):
            raise ValueError("lon/lat shapes must match LOS/HGT.")

    # LOS vector in local ENU (ISCE CCW azimuth)
    sE, sN, sU = los_enu_ground_to_sat_isce_ccw(inc, az)

    # Surface normal (unit) via central 3D differences in local ENU
    nE, nN, nU = surface_normal_from_llh_central(lon, lat, h)

    # Local incidence angle and cosine
    lia, costh = lia_and_costh_from_normal(nE, nN, nU, sE, sN, sU)

    # ALWAYS-ON masks:
    # Option A: shadow directly from cos(theta)<0 (avoids extra acos)
    shadow = (costh < 0).astype(np.uint8)
    layover = compute_layover_mask(nE, nN, nU, sE, sN, sU, lia, inc)

    # Write outputs
    write_geotiff_like(args.out, out_prof, lia, rasterio.float32, nodata=np.nan)
    write_geotiff_like(args.shadow_out, out_prof, shadow, rasterio.uint8, nodata=0)
    write_geotiff_like(args.layover_out, out_prof, layover, rasterio.uint8, nodata=0)

    print(f"Wrote: {args.out}")
    print(f"Wrote: {args.shadow_out}")
    print(f"Wrote: {args.layover_out}")


def run_lia(los_path, hgt_path, lon_path, lat_path,
            out_path="localIncAngle.tif",
            shadow_path="shadowMask.tif",
            layover_path="layoverMask.tif"):
    # This mirrors the code inside main(), but without argparse.
    import rasterio
    import numpy as np

    # Read inputs (same as in main)
    with rasterio.open(los_path) as ds_los:
        los = ds_los.read().astype(np.float32)
        inc, az = los[0], los[1]
        H, W = inc.shape
        out_prof = ds_los.profile

    with rasterio.open(hgt_path) as ds_hgt:
        h = ds_hgt.read(1).astype(np.float32)
        if h.shape != (H, W):
            raise ValueError("HGT shape must match LOS shape.")

    with rasterio.open(lon_path) as ds_lon, rasterio.open(lat_path) as ds_lat:
        lon = ds_lon.read(1).astype(np.float64)
        lat = ds_lat.read(1).astype(np.float64)
        if lon.shape != (H, W) or lat.shape != (H, W):
            raise ValueError("lon/lat shapes must match LOS/HGT.")

    # Use the script’s functions
    sE, sN, sU = los_enu_ground_to_sat_isce_ccw(inc, az)
    nE, nN, nU = surface_normal_from_llh_central(lon, lat, h)
    lia, costh = lia_and_costh_from_normal(nE, nN, nU, sE, sN, sU)

    shadow = (costh < 0).astype(np.uint8)
    layover = compute_layover_mask(nE, nN, nU, sE, sN, sU, lia, inc)

    # Write GeoTIFFs
    write_geotiff_like(out_path,        out_prof, lia,    rasterio.float32, nodata=np.nan)
    write_geotiff_like(shadow_path,     out_prof, shadow, rasterio.uint8,   nodata=0)
    write_geotiff_like(layover_path,    out_prof, layover,rasterio.uint8,   nodata=0)

    return lia, shadow, layover  # also return arrays for immediate use


if __name__ == "__main__":
    main()