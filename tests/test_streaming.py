"""
Tests for NISAR streaming and local-cache utilities.
"""

from __future__ import annotations

import io
import os
import sys
import types
from pathlib import Path

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
        download_with_progress,
        get_nisar_cache_path,
        open_nisar_h5_stream,
        search_nisar_data,
        setup_earthaccess_auth,
    )

    assert open_nisar_h5_stream is not None
    assert search_nisar_data is not None
    assert setup_earthaccess_auth is not None
    assert download_with_progress is not None
    assert cache_nisar_granule is not None
    assert get_nisar_cache_path is not None


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
        login=lambda persist=False: types.SimpleNamespace(authenticated=True),
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
        login=lambda persist=False: types.SimpleNamespace(authenticated=True),
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
        login=lambda persist=False: types.SimpleNamespace(authenticated=True),
    )
    monkeypatch.setitem(sys.modules, "earthaccess", fake_earthaccess)

    granule = FakeEarthaccessGranule("https://example.com/reused_granule.h5")
    cache_path = get_nisar_cache_path(granule, cache_dir=tmp_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_bytes(b"cached")

    resolved = cache_nisar_granule(granule, cache_dir=tmp_path)
    assert resolved == cache_path
    assert download_calls == []


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
        login=lambda persist=False: types.SimpleNamespace(authenticated=True),
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
