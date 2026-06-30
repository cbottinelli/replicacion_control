# monitoring/json_cache.py

import hashlib
import os
from typing import Any, Optional, Union

from cattrs.preconf.json import make_converter

from elk.func_correct.loaded_problem import ToEval


cache_dir = "../data"

json_converter = make_converter()

# Hook agregado por compatibilidad de versiones cattrs.
json_converter.register_structure_hook(
    Union[ToEval, Any],
    lambda v, _: v,
)


def cache_key(data: str, prefix: str) -> str:
    digest = hashlib.sha256(data.encode("utf-8")).hexdigest()
    return f"{prefix}:{digest}"


class JsonDiskCache:
    """
    Generic persistent key-value cache backed by a JSON file.
    Uses the shared project json_converter for serialization.
    """

    def __init__(
        self,
        cache_path: str,
        converter=json_converter,
        autosave: bool = False,
    ):
        self.cache_path = os.path.join(cache_dir, cache_path)
        self.converter = converter
        self.autosave = autosave
        self._cache: dict[str, Any] = {}
        self._loaded = False

    def _load(self) -> None:
        if self._loaded:
            return

        if os.path.exists(self.cache_path):
            with open(self.cache_path, "r") as f:
                text = f.read().strip()

            if text:
                self._cache = self.converter.loads(text, dict[str, Any])
            else:
                self._cache = {}

        self._loaded = True

    def dump(self) -> None:
        directory = os.path.dirname(self.cache_path)

        if directory:
            os.makedirs(directory, exist_ok=True)

        with open(self.cache_path, "w") as f:
            f.write(self.converter.dumps(self._cache))

    def get(self, key: str, default: Optional[Any] = None) -> Any:
        self._load()
        return self._cache.get(key, default)

    def set(self, key: str, value: Any, autosave: Optional[bool] = None) -> None:
        self._load()
        self._cache[key] = value

        should_save = self.autosave if autosave is None else autosave

        if should_save:
            self.dump()

    def contains(self, key: str) -> bool:
        self._load()
        return key in self._cache

    def clear(self) -> None:
        self._cache = {}
        self._loaded = True
        self.dump()