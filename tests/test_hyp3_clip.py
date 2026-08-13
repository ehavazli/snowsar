"""Tests for HyP3 product clipping functions."""
from __future__ import annotations

import numpy as np
import pytest
from pathlib import Path

pytest.importorskip("rasterio")


@pytest.fixture
def synthetic_raster(tmp_path):
    """Create a synthetic GeoTIFF with known bounds."""
    import rasterio
    from rasterio.transform import from_bounds

    def _create(
        filename: str,
        bounds: tuple[float, float, float, float],
        width: int = 100,
        height: int = 100,
        crs: str = "EPSG:32610",
    ) -> Path:
        """Create a test GeoTIFF with given bounds (left, bottom, right, top)."""
        left, bottom, right, top = bounds
        transform = from_bounds(left, bottom, right, top, width, height)

        data = np.random.randint(0, 255, (1, height, width), dtype=np.uint8)

        path = tmp_path / filename
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=height,
            width=width,
            count=1,
            dtype=np.uint8,
            crs=crs,
            transform=transform,
        ) as dst:
            dst.write(data)

        return path

    return _create


def test_common_geotiff_overlap_basic(synthetic_raster):
    """Test common overlap computation for two overlapping rasters."""
    from snowsar.utils import common_geotiff_overlap

    # Create two rasters with known overlap
    # Raster 1: [0, 0, 100, 100]
    raster1 = synthetic_raster("r1.tif", (0, 0, 100, 100))
    # Raster 2: [50, 50, 150, 150]
    raster2 = synthetic_raster("r2.tif", (50, 50, 150, 150))

    # Expected overlap: [50, 100, 100, 50] (left, top, right, bottom)
    overlap = common_geotiff_overlap([raster1, raster2])

    left, top, right, bottom = overlap
    assert left == pytest.approx(50.0)
    assert top == pytest.approx(100.0)
    assert right == pytest.approx(100.0)
    assert bottom == pytest.approx(50.0)


def test_common_geotiff_overlap_no_overlap(synthetic_raster):
    """Test that non-overlapping rasters raise ValueError."""
    from snowsar.utils import common_geotiff_overlap

    # Raster 1: [0, 0, 50, 50]
    raster1 = synthetic_raster("r1.tif", (0, 0, 50, 50))
    # Raster 2: [100, 100, 150, 150] - no overlap
    raster2 = synthetic_raster("r2.tif", (100, 100, 150, 150))

    with pytest.raises(ValueError, match="No overlap exists"):
        common_geotiff_overlap([raster1, raster2])


def test_common_geotiff_overlap_crs_mismatch(synthetic_raster):
    """Test that mismatched CRS raises ValueError."""
    from snowsar.utils import common_geotiff_overlap

    raster1 = synthetic_raster("r1.tif", (0, 0, 100, 100), crs="EPSG:32610")
    raster2 = synthetic_raster("r2.tif", (0, 0, 100, 100), crs="EPSG:4326")

    with pytest.raises(ValueError, match="CRS mismatch"):
        common_geotiff_overlap([raster1, raster2])


def test_common_geotiff_overlap_empty_list():
    """Test that empty file list raises ValueError."""
    from snowsar.utils import common_geotiff_overlap

    with pytest.raises(ValueError, match="cannot be empty"):
        common_geotiff_overlap([])


def test_clip_geotiff_to_bounds(synthetic_raster, tmp_path):
    """Test clipping a single GeoTIFF to bounds."""
    from snowsar.utils import clip_geotiff_to_bounds
    import rasterio

    # Create raster: [0, 0, 100, 100]
    raster = synthetic_raster("input.tif", (0, 0, 100, 100))

    # Clip to subset: [25, 75, 75, 25] (left, top, right, bottom)
    output = tmp_path / "clipped.tif"
    result = clip_geotiff_to_bounds(raster, output, (25, 75, 75, 25))

    assert result == output
    assert output.exists()

    # Verify output bounds
    with rasterio.open(output) as src:
        bounds = src.bounds
        # Allow small floating point tolerance
        assert bounds.left == pytest.approx(25.0, abs=1e-6)
        assert bounds.top == pytest.approx(75.0, abs=1e-6)
        assert bounds.right == pytest.approx(75.0, abs=1e-6)
        assert bounds.bottom == pytest.approx(25.0, abs=1e-6)


def test_clip_geotiff_no_overwrite(synthetic_raster, tmp_path):
    """Test that existing output without overwrite=True raises ValueError."""
    from snowsar.utils import clip_geotiff_to_bounds

    raster = synthetic_raster("input.tif", (0, 0, 100, 100))
    output = tmp_path / "clipped.tif"

    # Create output first time
    clip_geotiff_to_bounds(raster, output, (25, 75, 75, 25))

    # Try again without overwrite
    with pytest.raises(ValueError, match="exists and overwrite=False"):
        clip_geotiff_to_bounds(raster, output, (25, 75, 75, 25), overwrite=False)


def test_clip_hyp3_products_to_common_overlap(synthetic_raster, tmp_path):
    """Test end-to-end HyP3 product clipping workflow."""
    from snowsar.utils import clip_hyp3_products_to_common_overlap

    # Create mock HyP3 directory structure
    pair1_dir = tmp_path / "S1_20201215_20201227"
    pair2_dir = tmp_path / "S1_20201227_20210108"
    pair1_dir.mkdir()
    pair2_dir.mkdir()

    # Create overlapping DEMs
    synthetic_raster(
        f"{pair1_dir.name}/S1_20201215_20201227_dem.tif",
        (0, 0, 100, 100)
    )
    synthetic_raster(
        f"{pair2_dir.name}/S1_20201227_20210108_dem.tif",
        (50, 50, 150, 150)
    )

    # Create corresponding unw_phase files
    synthetic_raster(
        f"{pair1_dir.name}/S1_20201215_20201227_unw_phase.tif",
        (0, 0, 100, 100)
    )
    synthetic_raster(
        f"{pair2_dir.name}/S1_20201227_20210108_unw_phase.tif",
        (50, 50, 150, 150)
    )

    # Clip only unw_phase files
    outputs = clip_hyp3_products_to_common_overlap(
        tmp_path,
        dem_pattern="*/*_dem.tif",
        suffixes=["_unw_phase.tif"],
    )

    # Should have 2 clipped files
    assert len(outputs) == 2

    # Verify naming convention
    clipped_names = [p.name for p in outputs]
    assert "S1_20201215_20201227_unw_phase_clipped.tif" in clipped_names
    assert "S1_20201227_20210108_unw_phase_clipped.tif" in clipped_names

    # All outputs should exist
    for out in outputs:
        assert out.exists()


def test_clip_hyp3_products_no_dem_error(tmp_path):
    """Test that missing DEMs raise ValueError."""
    from snowsar.utils import clip_hyp3_products_to_common_overlap

    with pytest.raises(ValueError, match="No DEM files found"):
        clip_hyp3_products_to_common_overlap(tmp_path)
