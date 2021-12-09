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

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        filepath: str,
        version: Version = None,
        layer: str = None,
    ) -> None:
        """Creates a new instance of ``CSVDataSet`` pointing to a concrete CSV file
        on a specific filesystem.
        """

        protocol, path = get_protocol_and_path(filepath, version)

        self._protocol = protocol
        self._fs = fsspec.filesystem(self._protocol)

        super().__init__(
            filepath=PurePosixPath(path),
            version=version,
            exists_function=self._fs.exists,
            glob_function=self._fs.glob,
        )

        self._layer = layer

    def _describe(self) -> Dict[str, Any]:
        return dict(
            filepath=self._filepath,
            protocol=self._protocol,
            version=self._version,
            layer=self._layer,
        )

    def _load(self) -> pd.DataFrame:
        load_path = get_filepath_str(self._get_load_path(), self._protocol)
        return scipy.sparse.load_npz(load_path)

    def _save(self, data: pd.DataFrame) -> None:
        save_path = get_filepath_str(self._get_save_path(), self._protocol)
        scipy.sparse.save_npz(save_path, data)

    def _exists(self) -> bool:
        try:
            load_path = get_filepath_str(self._get_load_path(), self._protocol)
        except DataSetError:
            return False

        return self._fs.exists(load_path)
