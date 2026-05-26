from __future__ import annotations

import pickle
import tempfile
from pathlib import Path
from typing import Any, Union


def save_pickle(obj: Any, path: Union[str, Path]) -> None:
    """
    Save Python object to pickle file with highest protocol.

    Creates parent directories if they don't exist.

    Parameters
    ----------
    obj : Any
        Python object to serialize
    path : str or Path
        Output pickle file path

    Notes
    -----
    Uses pickle.HIGHEST_PROTOCOL for best performance and compatibility
    with newer Python versions.

    Examples
    --------
    >>> data = {"results": [1, 2, 3], "metadata": {"date": "2023-01-01"}}
    >>> save_pickle(data, "outputs/results.pkl")
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_name = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as f:
            tmp_name = f.name
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        Path(tmp_name).replace(path)
    finally:
        if tmp_name is not None:
            tmp_path = Path(tmp_name)
            if tmp_path.exists():
                tmp_path.unlink()


def load_pickle(path: Union[str, Path]) -> Any:
    """
    Load Python object from pickle file.

    Parameters
    ----------
    path : str or Path
        Path to pickle file

    Returns
    -------
    Any
        Deserialized Python object

    Raises
    ------
    FileNotFoundError
        If pickle file doesn't exist

    Warnings
    --------
    Pickle files can execute arbitrary code during deserialization.
    Only load pickle files from trusted sources.

    Examples
    --------
    >>> data = load_pickle("outputs/results.pkl")
    >>> data["results"]
    [1, 2, 3]
    """
    path = Path(path)
    with path.open("rb") as f:
        return pickle.load(f)
