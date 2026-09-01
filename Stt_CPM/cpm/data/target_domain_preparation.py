import math

import numpy as np
import torch

from .augmentation import apply_radiation_noise
from .hyperspectral_dataset import HyperspectralPatchDataset
from .spatial_patch_ops import embed_data_in_zero_padded_canvas


def _build_target_train_test_loaders(
    standardized_cube,
    ground_truth,
    class_count: int,
    target_labeled_samples_per_class: int,
    half_width: int,
):
    print(standardized_cube.shape)
    row_count, column_count, band_count = standardized_cube.shape

    padded_cube = embed_data_in_zero_padded_canvas(standardized_cube)
    padded_ground_truth = embed_data_in_zero_padded_canvas(ground_truth)

    target_ground_truth = padded_ground_truth[
        row_count - half_width: 2 * row_count + half_width,
        column_count - half_width: 2 * column_count + half_width,
    ]
    target_cube = padded_cube[
        row_count - half_width: 2 * row_count + half_width,
        column_count - half_width: 2 * column_count + half_width,
        :,
    ]

    row_indices, column_indices = np.nonzero(target_ground_truth)
    sample_count = np.size(row_indices)
    print("number of sample", sample_count)

    class_train_indices = {}
    class_test_indices = {}
    augmented_train_indices_by_class = {}

    label_budget = int(target_labeled_samples_per_class)
    print("labeled number per class:", label_budget)
    print((200 - label_budget) / label_budget + 1)
    print(math.ceil((200 - label_budget) / label_budget) + 1)

    class_total = int(np.max(target_ground_truth))
    if class_total != class_count:
        raise ValueError(
            f"Config class_count={class_count}, but ground truth contains "
            f"{class_total} non-background classes."
        )

    for class_id in range(class_total):
        indices = [
            sample_index
            for sample_index, _ in enumerate(row_indices.ravel().tolist())
            if target_ground_truth[
                row_indices[sample_index],
                column_indices[sample_index],
            ] == class_id + 1
        ]
        np.random.shuffle(indices)

        class_train_indices[class_id] = indices[:label_budget]
        augmented_train_indices_by_class[class_id] = []

        augmentation_repeats = (
            math.ceil((200 - label_budget) / label_budget) + 1
        )
        for _ in range(augmentation_repeats):
            augmented_train_indices_by_class[class_id] += indices[:label_budget]

        class_test_indices[class_id] = indices[label_budget:]

    train_indices = []
    test_indices = []
    augmented_train_indices = []
    for class_id in range(class_total):
        train_indices += class_train_indices[class_id]
        test_indices += class_test_indices[class_id]
        augmented_train_indices += augmented_train_indices_by_class[class_id]

    np.random.shuffle(test_indices)

    print("the number of train_indices:", len(train_indices))
    print("the number of test_indices:", len(test_indices))
    print(
        "the number of train_indices after data augmentation:",
        len(augmented_train_indices),
    )
    print("labeled sample indices:", train_indices)

    train_count = len(train_indices)
    test_count = len(test_indices)
    augmented_train_count = len(augmented_train_indices)

    imdb = {
        "data": np.zeros(
            [
                2 * half_width + 1,
                2 * half_width + 1,
                band_count,
                train_count + test_count,
            ],
            dtype=np.float32,
        ),
        "Labels": np.zeros(
            train_count + test_count,
            dtype=np.int64,
        ),
        "set": np.zeros(
            train_count + test_count,
            dtype=np.int64,
        ),
    }

    train_test_permutation = np.array(train_indices + test_indices)

    for sample_index in range(train_count + test_count):
        source_index = train_test_permutation[sample_index]
        imdb["data"][:, :, :, sample_index] = target_cube[
            row_indices[source_index] - half_width:
                row_indices[source_index] + half_width + 1,
            column_indices[source_index] - half_width:
                column_indices[source_index] + half_width + 1,
            :,
        ]
        imdb["Labels"][sample_index] = target_ground_truth[
            row_indices[source_index],
            column_indices[source_index],
        ].astype(np.int64)

    imdb["Labels"] -= 1
    imdb["set"] = np.hstack(
        (
            np.ones(train_count),
            3 * np.ones(test_count),
        )
    ).astype(np.int64)
    print("Target train/test patch extraction is complete.")

    train_dataset = HyperspectralPatchDataset(imdb, train=True)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=class_count * label_budget,
        shuffle=False,
    )

    test_dataset = HyperspectralPatchDataset(imdb, train=False)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=100,
        shuffle=False,
    )

    augmented_imdb = {
        "data": np.zeros(
            [
                2 * half_width + 1,
                2 * half_width + 1,
                band_count,
                augmented_train_count,
            ],
            dtype=np.float32,
        ),
        "Labels": np.zeros(
            augmented_train_count,
            dtype=np.int64,
        ),
        "set": np.ones(
            augmented_train_count,
            dtype=np.int64,
        ),
    }

    augmented_permutation = np.array(augmented_train_indices)

    for sample_index in range(augmented_train_count):
        source_index = augmented_permutation[sample_index]
        patch = target_cube[
            row_indices[source_index] - half_width:
                row_indices[source_index] + half_width + 1,
            column_indices[source_index] - half_width:
                column_indices[source_index] + half_width + 1,
            :,
        ]
        augmented_imdb["data"][:, :, :, sample_index] = (
            apply_radiation_noise(patch)
        )
        augmented_imdb["Labels"][sample_index] = target_ground_truth[
            row_indices[source_index],
            column_indices[source_index],
        ].astype(np.int64)

    augmented_imdb["Labels"] -= 1

    return (
        train_loader,
        test_loader,
        augmented_imdb,
        target_ground_truth,
        train_test_permutation,
        row_indices,
        column_indices,
        train_count,
    )


def prepare_target_domain_data(
    standardized_cube,
    ground_truth,
    class_count: int,
    target_labeled_samples_per_class: int,
    patch_size: int,
):
    (
        train_loader,
        test_loader,
        augmented_imdb,
        target_ground_truth,
        train_test_permutation,
        row_indices,
        column_indices,
        train_count,
    ) = _build_target_train_test_loaders(
        standardized_cube=standardized_cube,
        ground_truth=ground_truth,
        class_count=class_count,
        target_labeled_samples_per_class=target_labeled_samples_per_class,
        half_width=patch_size // 2,
    )

    augmented_data = np.transpose(
        augmented_imdb["data"],
        (3, 2, 0, 1),
    )
    augmented_labels = augmented_imdb["Labels"]

    target_meta_training_data = {}
    for class_label, patch in zip(augmented_labels, augmented_data):
        target_meta_training_data.setdefault(class_label, []).append(patch)

    return (
        train_loader,
        test_loader,
        target_meta_training_data,
        target_ground_truth,
        train_test_permutation,
        row_indices,
        column_indices,
        train_count,
    )
