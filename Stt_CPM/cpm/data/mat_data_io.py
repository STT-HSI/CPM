from pathlib import Path
from typing import Optional

import numpy as np
import scipy.io as sio
from sklearn import preprocessing


def _resolve_mat_variable(
    mat_contents,
    requested_key: Optional[str],
    file_path: str,
) -> np.ndarray:
    if requested_key and requested_key in mat_contents:
        return mat_contents[requested_key]

    candidate_keys = [
        key
        for key in mat_contents.keys()
        if not key.startswith("__")
    ]
    if len(candidate_keys) == 1:
        return mat_contents[candidate_keys[0]]

    available = ", ".join(candidate_keys)
    raise KeyError(
        f"Could not resolve MATLAB variable for {file_path}. "
        f"Requested key={requested_key!r}; available data keys=[{available}]."
    )


def load_standardized_mat_dataset(
    image_file: str,
    ground_truth_file: str,
    image_mat_key: Optional[str] = None,
    ground_truth_mat_key: Optional[str] = None,
):
    image_contents = sio.loadmat(image_file)
    ground_truth_contents = sio.loadmat(ground_truth_file)

    hyperspectral_cube = _resolve_mat_variable(
        image_contents,
        image_mat_key,
        image_file,
    )
    ground_truth = _resolve_mat_variable(
        ground_truth_contents,
        ground_truth_mat_key,
        ground_truth_file,
    )

    row_count, column_count, band_count = hyperspectral_cube.shape
    print(
        Path(image_file).stem,
        row_count,
        column_count,
        band_count,
    )

    flattened = hyperspectral_cube.reshape(
        np.prod(hyperspectral_cube.shape[:2]),
        np.prod(hyperspectral_cube.shape[2:]),
    )
    standardized = preprocessing.scale(flattened.astype(float))
    standardized_cube = standardized.reshape(hyperspectral_cube.shape)
    return standardized_cube, ground_truth
