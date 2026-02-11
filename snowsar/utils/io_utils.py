from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Union


def save_pickle(obj: Any, path: Union[str, Path]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(path: Union[str, Path]) -> Any:
    path = Path(path)
    with path.open("rb") as f:
        return pickle.load(f)
