"""
Tests for NISAR streaming and local-cache utilities.
"""

from __future__ import annotations

import io
import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")


def _make_stream_buffer() -> io.BytesIO:
    """Create a tiny in-memory HDF5 file for streaming tests."""
    buffer = io.BytesIO()
    with h5py.File(buffer, "w") as h5:
        science = h5.create_group("science")
        science.create_group("LSAR")
    buffer.seek(0)
    return buffer


def _write_local_h5(path: Path) -> Path:
    """Create a tiny local HDF5 file for cached-file tests."""
    with h5py.File(path, "w") as h5:
        science = h5.create_group("science")
        science.create_group("LSAR")
    return path


def _write_geogrid_h5(path: Path) -> Path:
    """Create a tiny lon/lat geogrid HDF5 file for bbox subset tests."""
    with h5py.File(path, "w") as h5:
        grp = h5.create_group("science/LSAR/GUNW/grids/frequencyA/test")
        grp.create_dataset("xCoordinates", data=[-120.5, -120.25, -120.0])
        grp.create_dataset("yCoordinates", data=[39.5, 39.25, 39.0, 38.75])
        grp.create_dataset("projection", data=4326)
        grp.create_dataset("layer", data=np.arange(12).reshape(4, 3))
    return path


def _write_stream_extract_h5(path: Path) -> Path:
    """Create a tiny GUNW-like file for streamed layer extraction tests."""
    with h5py.File(path, "w") as h5:
        grids = h5.create_group(
            "science/LSAR/GUNW/grids/frequencyA/unwrappedInterferogram/HH"
        )
        x = np.array([-120.5, -120.25, -120.0], dtype=float)
        y = np.array([39.5, 39.25, 39.0, 38.75], dtype=float)
        grids.create_dataset("xCoordinates", data=x)
        grids.create_dataset("yCoordinates", data=y)
        grids.create_dataset("projection", data=4326)

        unwrapped = np.arange(12, dtype=np.float32).reshape(4, 3)
        dset = grids.create_dataset("unwrappedPhase", data=unwrapped)
        dset.attrs["_FillValue"] = np.float32(-9999.0)
        grids.create_dataset(
            "coherenceMagnitude",
            data=np.linspace(0.1, 0.9, 12, dtype=np.float32).reshape(4, 3),
        )
        grids.create_dataset("connectedComponents", data=np.arange(12).reshape(4, 3))
        grids.create_dataset(
            "ionospherePhaseScreen",
            data=(unwrapped + 100).astype(np.float32),
        )

        radar = h5.create_group("science/LSAR/GUNW/metadata/radarGrid")
        radar.create_dataset("xCoordinates", data=x)
        radar.create_dataset("yCoordinates", data=y)
        radar.create_dataset("heightAboveEllipsoid", data=[0.0, 100.0])

        cube_shape = (2, 4, 3)
        radar.create_dataset(
            "losUnitVectorX", data=np.ones(cube_shape, dtype=np.float32) * 0.1
        )
        radar.create_dataset(
            "losUnitVectorY", data=np.ones(cube_shape, dtype=np.float32) * 0.2
        )
        radar.create_dataset(
            "incidenceAngle", data=np.ones(cube_shape, dtype=np.float32) * 30.0
        )
        radar.create_dataset(
            "hydrostaticTroposphericPhaseScreen",
            data=np.ones(cube_shape, dtype=np.float32) * 3.0,
        )
        radar.create_dataset(
            "wetTroposphericPhaseScreen",
            data=np.ones(cube_shape, dtype=np.float32) * 4.0,
        )
    return path


class FakeEarthaccessGranule:
    """Minimal earthaccess-style granule object."""

    def __init__(self, url: str = "https://example.com/NISAR_test.h5") -> None:
        self._url = url
        self.umm = {"GranuleUR": Path(url).name}

    def data_links(self, access=None, in_region: bool = False):
        return [self._url]


class FakeAsfResult:
    """Minimal ASF Search result object."""

    def __init__(self, url: str = "https://example.com/NISAR_asf.h5") -> None:
        self._url = url
        self.properties = {"fileName": Path(url).name}

    def get_urls(self):
        return [self._url]


def test_streaming_exports_are_public():
    """The public lazy API should expose the streaming helpers."""
    from snowsar.utils import (
        cache_nisar_granule,
        extract_gunw_layers_to_geotiff_bbox_streamed,
        download_with_progress,
        get_nisar_cache_path,
        get_nisar_granule_name,
        get_nisar_granule_urls,
        get_nisar_subset_cache_path,
        open_nisar_h5_stream,
        read_nisar_h5_bbox,
        read_nisar_h5_bbox_cached,
        search_nisar_data,
        setup_asf_search_auth,
        setup_earthaccess_auth,
        write_nisar_subset_geotiff,
    )

    assert open_nisar_h5_stream is not None
    assert search_nisar_data is not None
    assert setup_earthaccess_auth is not None
    assert setup_asf_search_auth is not None
    assert download_with_progress is not None
    assert cache_nisar_granule is not None
    assert extract_gunw_layers_to_geotiff_bbox_streamed is not None
    assert get_nisar_cache_path is not None
    assert get_nisar_granule_name is not None
    assert get_nisar_granule_urls is not None
    assert get_nisar_subset_cache_path is not None
    assert read_nisar_h5_bbox is not None
    assert read_nisar_h5_bbox_cached is not None
    assert write_nisar_subset_geotiff is not None


def test_granule_helpers_handle_earthaccess_results():
    """Display helpers should handle earthaccess-style granules."""
    from snowsar.utils.stream_utils import (
        get_nisar_granule_name,
        get_nisar_granule_urls,
    )

    granule = FakeEarthaccessGranule("https://example.com/earthaccess_file.h5")

    assert get_nisar_granule_name(granule) == "earthaccess_file.h5"
    assert get_nisar_granule_urls(granule) == [
        "https://example.com/earthaccess_file.h5"
    ]


def test_granule_helpers_handle_asf_search_results():
    """Display helpers should handle ASF Search result objects."""
    from snowsar.utils.stream_utils import (
        get_nisar_granule_name,
        get_nisar_granule_urls,
    )

    result = FakeAsfResult("https://example.com/asf_file.h5")

    assert get_nisar_granule_name(result) == "asf_file.h5"
    assert get_nisar_granule_urls(result) == ["https://example.com/asf_file.h5"]


def test_read_nisar_h5_bbox_subsets_geogrid(tmp_path):
    """BBox reads should return only the overlapping geogrid window."""
    from snowsar.utils.stream_utils import read_nisar_h5_bbox

    h5_path = _write_geogrid_h5(tmp_path / "geogrid.h5")
    dataset_path = "science/LSAR/GUNW/grids/frequencyA/test/layer"

    with h5py.File(h5_path, "r") as h5:
        data, x, y, epsg = read_nisar_h5_bbox(
            h5,
            dataset_path,
            bbox=(-120.4, 39.1, -120.1, 39.4),
        )

    assert epsg == 4326
    assert x.tolist() == [-120.25]
    assert y.tolist() == [39.25]
    assert data.tolist() == [[4]]


def test_read_nisar_h5_bbox_changes_with_extent(tmp_path):
    """Different bboxes should produce different subset shapes and values."""
    from snowsar.utils.stream_utils import read_nisar_h5_bbox

    h5_path = _write_geogrid_h5(tmp_path / "geogrid.h5")
    dataset_path = "science/LSAR/GUNW/grids/frequencyA/test/layer"

    with h5py.File(h5_path, "r") as h5:
        wide, _, _, _ = read_nisar_h5_bbox(
            h5,
            dataset_path,
            bbox=(-120.6, 38.9, -119.9, 39.6),
        )
        narrow, _, _, _ = read_nisar_h5_bbox(
            h5,
            dataset_path,
            bbox=(-120.4, 39.1, -120.1, 39.4),
        )

    assert wide.shape == (3, 3)
    assert narrow.shape == (1, 1)
    assert wide.tolist() != narrow.tolist()


def test_read_nisar_h5_bbox_cached_writes_subset_geotiff_only(tmp_path):
    """Subset cache should write a GeoTIFF, not a copied source HDF5 granule."""
    from snowsar.utils.stream_utils import read_nisar_h5_bbox_cached

    pytest.importorskip("rasterio")
    h5_path = _write_geogrid_h5(tmp_path / "geogrid.h5")
    cache_dir = tmp_path / "cache"
    dataset_path = "science/LSAR/GUNW/grids/frequencyA/test/layer"

    data, x, y, epsg, cache_path = read_nisar_h5_bbox_cached(
        h5_path,
        dataset_path,
        bbox=(-120.4, 39.1, -120.1, 39.4),
        cache_dir=cache_dir,
    )

    assert epsg == 4326
    assert data.tolist() == [[4]]
    assert x.tolist() == [-120.25]
    assert y.tolist() == [39.25]
    assert cache_path.suffix == ".tif"
    assert cache_path.exists()
    assert not (cache_dir / h5_path.name).exists()


def test_read_nisar_h5_bbox_cached_can_disable_cache(tmp_path):
    """Disabling cache should stream the subset without writing a cache file."""
    from snowsar.utils.stream_utils import read_nisar_h5_bbox_cached

    h5_path = _write_geogrid_h5(tmp_path / "geogrid.h5")
    cache_dir = tmp_path / "cache"
    dataset_path = "science/LSAR/GUNW/grids/frequencyA/test/layer"

    data, x, y, epsg, cache_path = read_nisar_h5_bbox_cached(
        h5_path,
        dataset_path,
        bbox=(-120.4, 39.1, -120.1, 39.4),
        cache_dir=cache_dir,
        use_cache=False,
    )

    assert epsg == 4326
    assert data.tolist() == [[4]]
    assert x.tolist() == [-120.25]
    assert y.tolist() == [39.25]
    assert cache_path is None
    assert not cache_dir.exists()


def test_write_nisar_subset_geotiff_writes_loaded_arrays(tmp_path):
    """Already-loaded subset arrays should be cacheable without reopening HDF5."""
    from snowsar.utils.stream_utils import write_nisar_subset_geotiff

    pytest.importorskip("rasterio")
    path = tmp_path / "subset.tif"

    resolved = write_nisar_subset_geotiff(
        path,
        np.array([[4]]),
        np.array([-120.25]),
        np.array([39.25]),
        4326,
    )

    assert resolved == path
    assert path.exists()


def test_read_nisar_h5_bbox_cached_reuses_existing_subset(monkeypatch, tmp_path):
    """Cache hits should not open the HDF5 source again."""
    from snowsar.utils import stream_utils

    pytest.importorskip("rasterio")
    granule = FakeEarthaccessGranule("https://example.com/reused_subset.h5")
    dataset_path = "science/LSAR/GUNW/grids/frequencyA/test/layer"
    bbox = (-120.4, 39.1, -120.1, 39.4)
    cache_path = stream_utils.get_nisar_subset_cache_path(
        granule,
        dataset_path,
        bbox,
        cache_dir=tmp_path,
    )
    stream_utils._write_subset_geotiff(
        cache_path,
        np.array([[9]]),
        np.array([-120.25]),
        np.array([39.25]),
        4326,
    )
    monkeypatch.setattr(
        stream_utils,
        "open_nisar_h5_stream",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("source should not be opened")
        ),
    )

    data, x, y, epsg, resolved_path = stream_utils.read_nisar_h5_bbox_cached(
        granule,
        dataset_path,
        bbox,
        cache_dir=tmp_path,
    )

    assert resolved_path == cache_path
    assert data.tolist() == [[9]]
    assert x.tolist() == [-120.25]
    assert y.tolist() == [39.25]
    assert epsg == 4326


def test_extract_gunw_layers_to_geotiff_bbox_streamed_reads_direct_layers(tmp_path):
    """Streamed bbox extraction should write direct geogrid layers without a full download."""
    from snowsar.utils.stream_utils import (
        _read_subset_geotiff,
        extract_gunw_layers_to_geotiff_bbox_streamed,
    )

    pytest.importorskip("rasterio")
    h5_path = _write_stream_extract_h5(tmp_path / "stream_extract.h5")

    outputs = extract_gunw_layers_to_geotiff_bbox_streamed(
        h5_path,
        bbox=(-120.4, 39.1, -120.1, 39.4),
        out_dir=tmp_path / "out",
        frequency="A",
        pol="HH",
        layers=["unwrappedPhase", "coherenceMagnitude", "connectedComponents"],
        provider="earthaccess",
    )

    assert sorted(outputs) == [
        "coherenceMagnitude",
        "connectedComponents",
        "unwrappedPhase",
    ]

    data, x, y, epsg = _read_subset_geotiff(outputs["unwrappedPhase"])
    assert epsg == 4326
    assert x.tolist() == [-120.25]
    assert y.tolist() == [39.25]
    assert data.tolist() == [[4.0]]

    cc, _, _, _ = _read_subset_geotiff(outputs["connectedComponents"])
    assert cc.shape == (1, 1)
    assert cc.dtype.kind == "f"


def test_extract_gunw_layers_to_geotiff_bbox_streamed_handles_cube_layers(
    monkeypatch, tmp_path
):
    """Cube and derived layers should route through the streamed interpolation helpers."""
    from snowsar.utils import stream_utils

    pytest.importorskip("rasterio")
    h5_path = _write_stream_extract_h5(tmp_path / "stream_extract_cube.h5")

    dem_calls = []

    def fake_download_dem(bbox, dem_out, **kwargs):
        dem_calls.append(("download", bbox, Path(dem_out)))
        Path(dem_out).parent.mkdir(parents=True, exist_ok=True)
        Path(dem_out).write_bytes(b"dem")
        return Path(dem_out)

    def fake_prepare_dem(*, x_out, y_out, out_epsg, **kwargs):
        dem_calls.append(("prepare", tuple(x_out), tuple(y_out), out_epsg))
        return np.ones((len(y_out), len(x_out)), dtype=np.float32)

    def fake_interp_inc_local(*args, x_out, y_out, **kwargs):
        shape = (len(y_out), len(x_out))
        return (
            np.full(shape, 11.0, dtype=np.float32),
            np.full(shape, 22.0, dtype=np.float32),
        )

    def fake_interp_cube_layer(*args, x_out, y_out, cube_ds_name, **kwargs):
        shape = (len(y_out), len(x_out))
        value = 33.0 if cube_ds_name == "losUnitVectorX" else 44.0
        return np.full(shape, value, dtype=np.float32)

    def fake_interp_cube_array(cube, *, x_out, y_out, **kwargs):
        shape = (len(y_out), len(x_out))
        return np.full(shape, float(np.nanmean(cube)), dtype=np.float32)

    monkeypatch.setattr(
        stream_utils,
        "_download_dem_for_bbox_with_sardem",
        fake_download_dem,
    )
    monkeypatch.setattr(stream_utils, "_prepare_bbox_dem_on_grid", fake_prepare_dem)
    monkeypatch.setattr(
        stream_utils,
        "_interpolate_incidence_and_local_to_bbox_grid",
        fake_interp_inc_local,
    )
    monkeypatch.setattr(
        stream_utils,
        "_interpolate_cube_layer_to_bbox_grid",
        fake_interp_cube_layer,
    )
    monkeypatch.setattr(
        stream_utils,
        "_interpolate_cube_array_to_bbox_grid",
        fake_interp_cube_array,
    )

    outputs = stream_utils.extract_gunw_layers_to_geotiff_bbox_streamed(
        h5_path,
        bbox=(-120.4, 39.1, -120.1, 39.4),
        out_dir=tmp_path / "out",
        frequency="A",
        pol="HH",
        layers=["losUnitVectorX", "localIncidenceAngle", "totalTroposphere"],
    )

    assert sorted(outputs) == [
        "localIncidenceAngle_interp",
        "losUnitVectorX_interp",
        "totalTroposphere_interp",
    ]
    assert any(call[0] == "download" for call in dem_calls)
    assert any(call[0] == "prepare" for call in dem_calls)


def test_extract_gunw_layers_to_geotiff_bbox_streamed_retries_transient_read_error(
    monkeypatch, tmp_path
):
    """Transient 5xx-style streaming failures should reopen and retry the extraction."""
    from snowsar.utils import stream_utils

    pytest.importorskip("rasterio")
    h5_path = _write_stream_extract_h5(tmp_path / "stream_retry.h5")

    calls = {"count": 0}

    class TransientError(Exception):
        status = 502

    original = stream_utils._read_valid_unwrapped_bbox

    def flaky_read(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise TransientError("Bad Gateway")
        return original(*args, **kwargs)

    monkeypatch.setattr(stream_utils, "_read_valid_unwrapped_bbox", flaky_read)

    outputs = stream_utils.extract_gunw_layers_to_geotiff_bbox_streamed(
        h5_path,
        bbox=(-120.4, 39.1, -120.1, 39.4),
        out_dir=tmp_path / "out",
        frequency="A",
        pol="HH",
        layers=["unwrappedPhase"],
        max_retries=1,
        retry_delay=0,
    )

    assert calls["count"] == 2
    assert "unwrappedPhase" in outputs


def test_setup_earthaccess_auth_uses_netrc_when_available(monkeypatch):
    """Earthaccess auth should prefer netrc credentials when present."""
    from snowsar.utils import stream_utils

    received = {}

    def fake_login(**kwargs):
        received.update(kwargs)
        return types.SimpleNamespace(authenticated=True)

    fake_earthaccess = types.SimpleNamespace(
        auth_environ=lambda: (_ for _ in ()).throw(RuntimeError("not authenticated")),
        login=fake_login,
    )
    monkeypatch.setitem(sys.modules, "earthaccess", fake_earthaccess)
    monkeypatch.setattr(
        stream_utils,
        "_earthdata_credentials_from_netrc",
        lambda: ("user", "password"),
    )

    stream_utils.setup_earthaccess_auth(persist=True)

    assert received == {"strategy": "netrc", "persist": True}


def test_setup_earthaccess_auth_prompts_when_netrc_missing(monkeypatch):
    """Earthaccess auth should prompt when no netrc credentials exist."""
    from snowsar.utils import stream_utils

    received = {}

    def fake_login(**kwargs):
        received.update(kwargs)
        return types.SimpleNamespace(authenticated=True)

    fake_earthaccess = types.SimpleNamespace(
        auth_environ=lambda: (_ for _ in ()).throw(RuntimeError("not authenticated")),
        login=fake_login,
    )
    monkeypatch.setitem(sys.modules, "earthaccess", fake_earthaccess)
    monkeypatch.setattr(stream_utils, "_earthdata_credentials_from_netrc", lambda: None)

    stream_utils.setup_earthaccess_auth(persist=True)

    assert received == {"strategy": "interactive", "persist": True}


def test_setup_asf_search_auth_uses_netrc_when_available(monkeypatch):
    """ASF Search auth should prefer Earthdata credentials from netrc."""
    from snowsar.utils import stream_utils

    received = {}
    session = object()
    monkeypatch.setattr(stream_utils, "_ASF_SEARCH_SESSION", None)

    class FakeAsfSession:
        def auth_with_creds(self, username, password):
            received["username"] = username
            received["password"] = password
            return session

    fake_asf = types.SimpleNamespace(ASFSession=FakeAsfSession)
    monkeypatch.setitem(sys.modules, "asf_search", fake_asf)
    monkeypatch.setattr(
        stream_utils,
        "_earthdata_credentials_from_netrc",
        lambda: ("user", "password"),
    )
    monkeypatch.setattr(
        stream_utils,
        "_prompt_earthdata_credentials",
        lambda: (_ for _ in ()).throw(AssertionError("prompt should not be used")),
    )

    resolved = stream_utils.setup_asf_search_auth()

    assert resolved is session
    assert received == {"username": "user", "password": "password"}


def test_setup_asf_search_auth_prompts_when_netrc_missing(monkeypatch):
    """ASF Search auth should prompt when no netrc credentials exist."""
    from snowsar.utils import stream_utils

    received = {}
    session = object()
    monkeypatch.setattr(stream_utils, "_ASF_SEARCH_SESSION", None)

    class FakeAsfSession:
        def auth_with_creds(self, username, password):
            received["username"] = username
            received["password"] = password
            return session

    fake_asf = types.SimpleNamespace(ASFSession=FakeAsfSession)
    monkeypatch.setitem(sys.modules, "asf_search", fake_asf)
    monkeypatch.setattr(stream_utils, "_earthdata_credentials_from_netrc", lambda: None)
    monkeypatch.setattr(
        stream_utils,
        "_prompt_earthdata_credentials",
        lambda: ("prompt_user", "prompt_password"),
    )

    resolved = stream_utils.setup_asf_search_auth()

    assert resolved is session
    assert received == {"username": "prompt_user", "password": "prompt_password"}


def test_open_nisar_h5_stream_accepts_earthaccess_style_granule(monkeypatch):
    """earthaccess granules should be passed through directly to earthaccess.open."""
    from snowsar.utils.stream_utils import open_nisar_h5_stream

    opened_targets = []

    def fake_open(granules, **kwargs):
        opened_targets.extend(granules)
        return [_make_stream_buffer()]

    fake_earthaccess = types.SimpleNamespace(
        auth_environ=lambda: True,
        open=fake_open,
        login=lambda *args, **kwargs: types.SimpleNamespace(authenticated=True),
    )
    monkeypatch.setitem(sys.modules, "earthaccess", fake_earthaccess)

    granule = FakeEarthaccessGranule()
    with open_nisar_h5_stream(granule) as h5:
        assert "science" in h5

    assert opened_targets == [granule]


def test_open_nisar_h5_stream_normalizes_asf_results(monkeypatch):
    """ASF Search results should be normalized to their first data URL."""
    from snowsar.utils.stream_utils import open_nisar_h5_stream

    opened_targets = []

    def fake_open(granules, **kwargs):
        opened_targets.extend(granules)
        return [_make_stream_buffer()]

    fake_earthaccess = types.SimpleNamespace(
        auth_environ=lambda: True,
        open=fake_open,
        login=lambda *args, **kwargs: types.SimpleNamespace(authenticated=True),
    )
    monkeypatch.setitem(sys.modules, "earthaccess", fake_earthaccess)

    result = FakeAsfResult("https://example.com/NISAR_from_asf.h5")
    with open_nisar_h5_stream(result) as h5:
        assert "science" in h5

    assert opened_targets == ["https://example.com/NISAR_from_asf.h5"]


def test_open_nisar_h5_stream_opens_local_path_without_earthaccess(
    monkeypatch, tmp_path
):
    """Cached local files should open directly without Earthdata auth."""
    from snowsar.utils.stream_utils import open_nisar_h5_stream

    local_h5 = _write_local_h5(tmp_path / "cached_local.h5")

    fake_earthaccess = types.SimpleNamespace(
        auth_environ=lambda: (_ for _ in ()).throw(
            AssertionError("earthaccess should not be used for local files")
        ),
        open=lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("earthaccess.open should not be used for local files")
        ),
        login=lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("earthaccess.login should not be used for local files")
        ),
    )
    monkeypatch.setitem(sys.modules, "earthaccess", fake_earthaccess)

    with open_nisar_h5_stream(str(local_h5)) as h5:
        assert "science" in h5


def test_cache_nisar_granule_reuses_existing_file(monkeypatch, tmp_path):
    """Caching should reuse an existing local file instead of downloading again."""
    from snowsar.utils.stream_utils import cache_nisar_granule, get_nisar_cache_path

    download_calls = []
    fake_earthaccess = types.SimpleNamespace(
        auth_environ=lambda: True,
        download=lambda *args, **kwargs: download_calls.append((args, kwargs)),
        login=lambda *args, **kwargs: types.SimpleNamespace(authenticated=True),
    )
    monkeypatch.setitem(sys.modules, "earthaccess", fake_earthaccess)

    granule = FakeEarthaccessGranule("https://example.com/reused_granule.h5")
    cache_path = get_nisar_cache_path(granule, cache_dir=tmp_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_bytes(b"cached")

    resolved = cache_nisar_granule(granule, cache_dir=tmp_path)
    assert resolved == cache_path
    assert download_calls == []


def test_cache_nisar_granule_reuses_existing_file_url(monkeypatch, tmp_path):
    """file:// URLs to local HDF5 files should use the local cache fast path."""
    from snowsar.utils.stream_utils import cache_nisar_granule, get_nisar_cache_path

    fake_earthaccess = types.SimpleNamespace(
        auth_environ=lambda: (_ for _ in ()).throw(
            AssertionError("earthaccess should not be used for local files")
        ),
        download=lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("earthaccess.download should not be used for local files")
        ),
        login=lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("earthaccess.login should not be used for local files")
        ),
        open=lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("earthaccess.open should not be used for local files")
        ),
    )
    monkeypatch.setitem(sys.modules, "earthaccess", fake_earthaccess)

    local_h5 = _write_local_h5(tmp_path / "source_file_url.h5")
    granule = local_h5.as_uri()

    cache_path = get_nisar_cache_path(granule, cache_dir=tmp_path / "cache")
    resolved = cache_nisar_granule(granule, cache_dir=tmp_path / "cache")

    assert cache_path == tmp_path / "cache" / local_h5.name
    assert resolved == cache_path
    assert resolved.read_bytes() == local_h5.read_bytes()


def test_cache_nisar_granule_reuses_existing_tilde_path(monkeypatch, tmp_path):
    """~/ paths should be expanded before deciding whether to reuse a local file."""
    from snowsar.utils.stream_utils import cache_nisar_granule, get_nisar_cache_path

    fake_earthaccess = types.SimpleNamespace(
        auth_environ=lambda: (_ for _ in ()).throw(
            AssertionError("earthaccess should not be used for local files")
        ),
        download=lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("earthaccess.download should not be used for local files")
        ),
        login=lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("earthaccess.login should not be used for local files")
        ),
        open=lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("earthaccess.open should not be used for local files")
        ),
    )
    monkeypatch.setitem(sys.modules, "earthaccess", fake_earthaccess)

    fake_home = tmp_path / "home"
    fake_home.mkdir()
    local_h5 = _write_local_h5(fake_home / "tilde_source.h5")
    monkeypatch.setenv("HOME", str(fake_home))

    granule = f"~/{local_h5.name}"
    cache_path = get_nisar_cache_path(granule, cache_dir=tmp_path / "cache")
    resolved = cache_nisar_granule(granule, cache_dir=tmp_path / "cache")

    assert cache_path == tmp_path / "cache" / local_h5.name
    assert resolved == cache_path
    assert resolved.read_bytes() == local_h5.read_bytes()


def test_cache_nisar_granule_missing_local_path_falls_back_to_download(
    monkeypatch, tmp_path
):
    """Missing local-looking paths should continue through the download flow."""
    from snowsar.utils import stream_utils

    granule = f"~/{tmp_path.name}_missing.h5"
    expected_cache_path = stream_utils.get_nisar_cache_path(
        granule, cache_dir=tmp_path / "cache"
    )
    download_calls = []

    def fake_download(granule_results, *, output_dir, provider, overwrite):
        download_calls.append(
            {
                "granule_results": granule_results,
                "output_dir": output_dir,
                "provider": provider,
                "overwrite": overwrite,
            }
        )
        return [expected_cache_path]

    monkeypatch.setattr(stream_utils, "download_with_progress", fake_download)

    resolved = stream_utils.cache_nisar_granule(granule, cache_dir=tmp_path / "cache")

    assert resolved == expected_cache_path
    assert download_calls == [
        {
            "granule_results": [granule],
            "output_dir": expected_cache_path.parent,
            "provider": "earthaccess",
            "overwrite": False,
        }
    ]


def test_download_with_progress_normalizes_earthaccess_inputs(monkeypatch, tmp_path):
    """Downloads should preserve DataGranule objects and normalize ASF results to URLs."""
    from snowsar.utils.stream_utils import download_with_progress

    received = {}

    def fake_download(granules, output_dir, **kwargs):
        received["granules"] = granules
        received["output_dir"] = output_dir
        return [Path(output_dir) / "downloaded.h5"]

    fake_earthaccess = types.SimpleNamespace(
        auth_environ=lambda: True,
        download=fake_download,
        login=lambda *args, **kwargs: types.SimpleNamespace(authenticated=True),
    )
    monkeypatch.setitem(sys.modules, "earthaccess", fake_earthaccess)

    granule = FakeEarthaccessGranule("https://example.com/ea_granule.h5")
    asf_result = FakeAsfResult("https://example.com/asf_granule.h5")

    paths = download_with_progress(
        [granule, asf_result],
        output_dir=tmp_path,
        provider="earthaccess",
    )

    assert paths == [tmp_path / "downloaded.h5"]
    assert received["granules"][0] is granule
    assert received["granules"][1] == "https://example.com/asf_granule.h5"


def test_download_with_progress_authenticates_asf_search(monkeypatch, tmp_path):
    """ASF Search downloads should use the authenticated ASF session helper."""
    from snowsar.utils import stream_utils

    received = {}
    session = object()

    class DownloadableAsfResult(FakeAsfResult):
        def download(self, path, session):
            received["path"] = path
            received["session"] = session
            (Path(path) / "NISAR_asf.h5").write_bytes(b"downloaded")

    monkeypatch.setitem(sys.modules, "asf_search", types.SimpleNamespace())
    monkeypatch.setattr(stream_utils, "setup_asf_search_auth", lambda: session)

    paths = stream_utils.download_with_progress(
        [DownloadableAsfResult()],
        output_dir=tmp_path,
        provider="asf_search",
    )

    assert paths == [tmp_path / "NISAR_asf.h5"]
    assert received == {"path": str(tmp_path), "session": session}


def test_download_with_progress_returns_existing_asf_file_when_skipping(
    monkeypatch, tmp_path
):
    """ASF Search downloads should report existing files when overwrite is false."""
    from snowsar.utils import stream_utils

    class DownloadableAsfResult(FakeAsfResult):
        def download(self, path, session):
            raise AssertionError("download should not be called for existing files")

    existing = tmp_path / "NISAR_asf.h5"
    existing.write_bytes(b"existing")

    monkeypatch.setitem(sys.modules, "asf_search", types.SimpleNamespace())
    monkeypatch.setattr(stream_utils, "setup_asf_search_auth", lambda: object())

    paths = stream_utils.download_with_progress(
        [DownloadableAsfResult()],
        output_dir=tmp_path,
        provider="asf_search",
    )

    assert paths == [existing]


def test_open_nisar_h5_stream_rejects_unvalidated_provider():
    """Only the validated earthaccess provider should be accepted for streaming."""
    from snowsar.utils.stream_utils import open_nisar_h5_stream

    with pytest.raises(NotImplementedError, match="earthaccess"):
        open_nisar_h5_stream("https://example.com/file.h5", provider="fsspec")


def test_search_nisar_data_rejects_invalid_provider():
    """Unknown search providers should raise a clear error."""
    from snowsar.utils.stream_utils import search_nisar_data

    with pytest.raises(ValueError, match="Unknown provider"):
        search_nisar_data(processing_level="GUNW", provider="invalid")


def test_search_nisar_data_rejects_flight_direction_for_earthaccess():
    """Earthaccess searches should fail fast on unsupported flight direction."""
    from snowsar.utils.stream_utils import search_nisar_data

    with pytest.raises(ValueError, match="provider='earthaccess'"):
        search_nisar_data(
            processing_level="GUNW",
            provider="earthaccess",
            flight_direction="DESCENDING",
        )


def test_search_nisar_data_converts_bbox_for_asf_search(monkeypatch):
    """ASF Search expects bbox geometry as intersectsWith WKT."""
    from snowsar.utils import stream_utils

    received = {}
    session = object()

    class FakeSearchOptions:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    def fake_search(opts):
        received.update(opts.kwargs)
        return ["result"]

    fake_asf = types.SimpleNamespace(
        ASFSearchOptions=FakeSearchOptions,
        search=fake_search,
    )
    monkeypatch.setitem(sys.modules, "asf_search", fake_asf)
    monkeypatch.setattr(stream_utils, "setup_asf_search_auth", lambda: session)

    results = stream_utils.search_nisar_data(
        bbox=(-120.5, 37.0, -118.5, 38.5),
        start_date="2026-01-01",
        end_date="2026-03-31",
        processing_level="GCOV",
        provider="asf_search",
        flight_direction="descending",
        max_results=1,
    )

    assert results == ["result"]
    assert received == {
        "dataset": "NISAR",
        "processingLevel": "GCOV",
        "maxResults": 1,
        "session": session,
        "start": "2026-01-01",
        "end": "2026-03-31",
        "flightDirection": "DESCENDING",
        "intersectsWith": (
            "POLYGON((-120.5 37.0, -118.5 37.0, -118.5 38.5, "
            "-120.5 38.5, -120.5 37.0))"
        ),
    }


def test_search_nisar_data_includes_private_gunw_asf_collection(monkeypatch):
    """Authenticated ASF Search should include public and private GUNW collections."""
    from snowsar.utils import stream_utils

    received = {}
    session = object()

    class FakeSearchOptions:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    def fake_search(opts):
        received.update(opts.kwargs)
        return ["result"]

    fake_asf = types.SimpleNamespace(
        ASFSearchOptions=FakeSearchOptions,
        search=fake_search,
    )
    monkeypatch.setitem(sys.modules, "asf_search", fake_asf)
    monkeypatch.setattr(stream_utils, "setup_asf_search_auth", lambda: session)

    results = stream_utils.search_nisar_data(
        processing_level="GUNW",
        provider="asf_search",
        max_results=1,
    )

    assert results == ["result"]
    assert received["session"] is session
    assert received["collections"] == [
        "C2850261892-ASF",
        "C4052499921-ASF",
    ]


def test_search_nisar_data_rejects_invalid_flight_direction(monkeypatch):
    """ASF Search flight direction should be ascending or descending."""
    from snowsar.utils.stream_utils import search_nisar_data

    fake_asf = types.SimpleNamespace(search=lambda **kwargs: [])
    monkeypatch.setitem(sys.modules, "asf_search", fake_asf)

    with pytest.raises(ValueError, match="flight_direction"):
        search_nisar_data(
            processing_level="GUNW",
            provider="asf_search",
            flight_direction="sideways",
        )


@pytest.mark.parametrize(
    ("processing_level", "short_name"),
    [
        ("RSLC", "NISAR_L1_RSLC_BETA_V1"),
        ("RIFG", "NISAR_L1_RIFG_BETA_V1"),
        ("ROFF", "NISAR_L1_ROFF_BETA_V1"),
        ("RUNW", "NISAR_L1_RUNW_BETA_V1"),
        ("GSLC", "NISAR_L2_GSLC_BETA_V1"),
        ("GOFF", "NISAR_L2_GOFF_BETA_V1"),
        ("GCOV", "NISAR_L2_GCOV_BETA_V1"),
        ("GUNW", "NISAR_L2_GUNW_BETA_V1"),
        ("SME2", "NISAR_L3_SME2_BETA_V1"),
    ],
)
def test_search_nisar_data_maps_all_public_earthaccess_products(
    monkeypatch, processing_level, short_name
):
    """Earthaccess searches should accept all public NISAR product acronyms."""
    from snowsar.utils.stream_utils import search_nisar_data

    received = {}

    def fake_search_data(**kwargs):
        received.update(kwargs)
        return ["result"]

    fake_earthaccess = types.SimpleNamespace(search_data=fake_search_data)
    monkeypatch.setitem(sys.modules, "earthaccess", fake_earthaccess)

    results = search_nisar_data(
        processing_level=processing_level,
        provider="earthaccess",
        max_results=1,
    )

    assert results == ["result"]
    assert received["short_name"] == short_name


@pytest.mark.skipif(
    os.getenv("SNOWSAR_RUN_EARTHACCESS_TESTS") != "1",
    reason=(
        "Set SNOWSAR_RUN_EARTHACCESS_TESTS=1 and configure Earthdata auth "
        "to run remote streaming integration."
    ),
)
def test_search_and_stream_integration():
    """
    Optional remote integration test for search -> stream -> dataset access.
    """
    from snowsar.utils import open_nisar_h5_stream, search_nisar_data

    results = search_nisar_data(
        bbox=(-120, 37, -119, 38),
        start_date="2026-01-01",
        end_date="2026-03-31",
        processing_level="GUNW",
        provider="earthaccess",
        max_results=1,
    )
    if not results:
        pytest.skip("No NISAR data found for test region/dates")

    with open_nisar_h5_stream(results[0]) as h5:
        assert "science" in h5
