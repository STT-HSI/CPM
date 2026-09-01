import numpy as np


def embed_data_in_zero_padded_canvas(data: np.ndarray) -> np.ndarray:
    zeros = np.zeros_like(data)
    top = np.concatenate((zeros, zeros, zeros), axis=1)
    middle = np.concatenate((zeros, data, zeros), axis=1)
    bottom = top
    return np.concatenate((top, middle, bottom), axis=0)


def zero_pad_hyperspectral_cube(
    matrix: np.ndarray,
    spatial_padding: int,
    spectral_padding: int = 0,
) -> np.ndarray:
    return np.pad(
        matrix,
        (
            (spatial_padding, spatial_padding),
            (spatial_padding, spatial_padding),
            (spectral_padding, spectral_padding),
        ),
        mode="constant",
        constant_values=0,
    )


def map_flat_indices_to_patch_coordinates(
    flat_indices,
    row_count: int,
    column_count: int,
    spatial_padding: int,
):
    del row_count
    assignments = {}
    for counter, value in enumerate(flat_indices):
        assignments[counter] = [
            value // column_count + spatial_padding,
            value % column_count + spatial_padding,
        ]
    return assignments


def extract_spatial_patch(
    matrix: np.ndarray,
    row: int,
    column: int,
    half_width: int,
) -> np.ndarray:
    selected_rows = matrix[
        range(row - half_width, row + half_width + 1),
        :,
    ]
    return selected_rows[
        :,
        range(column - half_width, column + half_width + 1),
    ]
