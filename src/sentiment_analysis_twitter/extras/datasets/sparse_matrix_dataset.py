"""``SparseMatrixDataSet`` loads/saves data from/to a npz file using scipy."""
from pathlib import PurePosixPath
from typing import Any, Dict

import fsspec
import pandas as pd
import scipy
from kedro.io.core import (
    AbstractVersionedDataSet,
    DataSetError,
    Version,
    get_filepath_str,
    get_protocol_and_path,
)


class SparseMatrixDataSet(AbstractVersionedDataSet):
    """``SparseMatrixDataSet`` loads/saves data from/to a npz file using scipy."""

    def __init__(
        self, filepath: str, version: Version | None = None, layer: str | None = None
    ) -> None:
        """Creates a new instance of ``SparseMatrixDataSet`` pointing to a specific file on a filesystem."""

        protocol, path = get_protocol_and_path(filepath, version)  # type: ignore
        self._fs = fsspec.filesystem(protocol)
        super().__init__(filepath=PurePosixPath(path), version=version)
        self._layer = layer
        self._load_path: str | None = None
        self._save_path: str | None = None
        self._protocol = protocol

    def _describe(self) -> Dict[str, Any]:
        return dict(
            filepath=self._filepath,
            protocol=self._protocol,
            version=self._version,
            layer=self._layer,
        )

    def _load(self) -> pd.DataFrame:
        if not self._load_path:
            self._load_path = get_filepath_str(self._get_load_path(), self._protocol)
        return scipy.sparse.load_npz(self._load_path)

    def _save(self, data: pd.DataFrame) -> None:
        if not self._save_path:
            self._save_path = get_filepath_str(self._get_save_path(), self._protocol)
        scipy.sparse.save_npz(self._save_path, data)

    def _exists(self) -> bool:
        return self._fs.exists(self._load_path) if self._load_path else False
