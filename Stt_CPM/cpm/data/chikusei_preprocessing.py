import argparse
import os
import pickle
from pathlib import Path

import h5py
import numpy as np
import scipy.io
from sklearn import preprocessing

from cpm.data.spatial_patch_ops import (
    extract_spatial_patch,
    map_flat_indices_to_patch_coordinates,
    zero_pad_hyperspectral_cube,
)


def sample_labeled_pixel_indices(ground_truth: np.ndarray):
    class_locations = {}
    class_count = int(max(ground_truth))
    for class_index in range(class_count):
        indices = [
            index
            for index, value in enumerate(ground_truth.ravel().tolist())
            if value == class_index + 1
        ]
        np.random.shuffle(indices)
        class_locations[class_index] = indices

    all_indices = []
    for class_index in range(class_count):
        all_indices += class_locations[class_index]

    np.random.shuffle(all_indices)
    return all_indices


def load_chikusei_image_data(image_file: str) -> np.ndarray:
    print(f"Loading image file: {image_file}")
    with h5py.File(image_file, "r") as hdf5_file:
        data = np.array(hdf5_file["chikusei"][:])
        print(f"Raw image data shape: {data.shape}")
        return data


def load_chikusei_ground_truth(ground_truth_file: str) -> np.ndarray:
    print(f"Loading label file: {ground_truth_file}")
    label_contents = scipy.io.loadmat(ground_truth_file)
    ground_truth_struct = label_contents["GT"]

    if ground_truth_struct.dtype.names is not None:
        if "gt" in ground_truth_struct.dtype.names:
            ground_truth = ground_truth_struct["gt"][0, 0]
        else:
            first_field = ground_truth_struct.dtype.names[0]
            ground_truth = ground_truth_struct[first_field][0, 0]
    else:
        ground_truth = ground_truth_struct

    if ground_truth.shape == (2517, 2335):
        ground_truth = ground_truth.T

    return ground_truth


def load_and_standardize_chikusei(
    image_file: str,
    ground_truth_file: str,
):
    data = load_chikusei_image_data(image_file)
    ground_truth = load_chikusei_ground_truth(ground_truth_file)

    if data.ndim != 3:
        raise ValueError(f"Unexpected image data shape: {data.shape}")

    data = np.transpose(data, (1, 2, 0))
    row_count, column_count, band_count = data.shape

    if ground_truth.shape != (row_count, column_count):
        if (
            ground_truth.shape[0] >= row_count
            and ground_truth.shape[1] >= column_count
        ):
            ground_truth = ground_truth[:row_count, :column_count]
        else:
            padded_ground_truth = np.zeros(
                (row_count, column_count),
                dtype=ground_truth.dtype,
            )
            valid_rows = min(row_count, ground_truth.shape[0])
            valid_columns = min(column_count, ground_truth.shape[1])
            padded_ground_truth[:valid_rows, :valid_columns] = (
                ground_truth[:valid_rows, :valid_columns]
            )
            ground_truth = padded_ground_truth

    flattened_ground_truth = ground_truth.reshape(-1)
    flattened_data = data.reshape(
        np.prod(data.shape[:2]),
        np.prod(data.shape[2:]),
    )
    standardized_data = preprocessing.scale(flattened_data)
    standardized_cube = standardized_data.reshape(
        row_count,
        column_count,
        band_count,
    )
    return standardized_cube, flattened_ground_truth


def build_chikusei_source_dataset(
    image_file: str,
    ground_truth_file: str,
    patch_half_width: int,
):
    standardized_cube, flattened_ground_truth = load_and_standardize_chikusei(
        image_file,
        ground_truth_file,
    )

    row_count, column_count, band_count = standardized_cube.shape
    padded_cube = zero_pad_hyperspectral_cube(
        standardized_cube,
        patch_half_width,
    )

    np.random.seed(1334)
    all_indices = sample_labeled_pixel_indices(flattened_ground_truth)
    sample_count = len(all_indices)

    patches = np.zeros(
        (
            sample_count,
            2 * patch_half_width + 1,
            2 * patch_half_width + 1,
            band_count,
        )
    )
    labels = flattened_ground_truth[all_indices] - 1

    assignments = map_flat_indices_to_patch_coordinates(
        all_indices,
        row_count,
        column_count,
        patch_half_width,
    )

    for sample_index in range(len(assignments)):
        patches[sample_index] = extract_spatial_patch(
            padded_cube,
            assignments[sample_index][0],
            assignments[sample_index][1],
            patch_half_width,
        )

    imdb = {
        "data": patches.astype(np.float32),
        "Labels": labels.astype(np.int64),
        "set": np.ones(sample_count, dtype=np.int64),
    }
    return imdb


def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess the Chikusei source-domain dataset for CPM."
    )
    parser.add_argument(
        "--image_file",
        default="datasets/chikusei/chikusei_hyperspectral_image.mat",
    )
    parser.add_argument(
        "--ground_truth_file",
        default="datasets/chikusei/chikusei_ground_truth.mat",
    )
    parser.add_argument(
        "--output_file",
        default="datasets/chikusei/chikusei_source_patches_128bands_7x7.pkl",
    )
    parser.add_argument("--patch_half_width", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    imdb = build_chikusei_source_dataset(
        args.image_file,
        args.ground_truth_file,
        args.patch_half_width,
    )

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        pickle.dump(imdb, handle, protocol=4)

    print(f"Source patches saved to {output_path}")
    print(f"Data shape: {imdb['data'].shape}")
    print(f"Labels shape: {imdb['Labels'].shape}")
    print(f"Unique labels: {np.unique(imdb['Labels'])}")


if __name__ == "__main__":
    main()
