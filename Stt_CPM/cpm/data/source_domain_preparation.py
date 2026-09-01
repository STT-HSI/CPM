import os
import pickle

import numpy as np


def select_source_classes_with_sufficient_samples(
    class_indexed_data,
    minimum_samples_per_class: int = 200,
):
    selected = {}
    class_count = 0
    sample_count = 0

    for class_id in class_indexed_data:
        if len(class_indexed_data[class_id]) >= minimum_samples_per_class:
            selected[class_id] = class_indexed_data[class_id][
                -minimum_samples_per_class:
            ]
            class_count += 1
            sample_count += len(selected[class_id])

    print("the number of class:", class_count)
    print("the number of sample:", sample_count)
    return selected


def load_source_meta_training_data(
    dataset_root: str,
    source_dataset_file: str,
):
    with open(
        os.path.join(dataset_root, source_dataset_file),
        "rb",
    ) as handle:
        source_imdb = pickle.load(handle)

    source_data = source_imdb["data"]
    source_labels = source_imdb["Labels"]

    unique_labels = sorted(list(set(source_labels)))
    label_encoder = {
        unique_labels[index]: index
        for index in range(len(unique_labels))
    }

    class_indexed_data = {}
    for class_label, patch in zip(source_labels, source_data):
        encoded_label = label_encoder[class_label]
        class_indexed_data.setdefault(encoded_label, []).append(patch)

    meta_training_data = select_source_classes_with_sufficient_samples(
        class_indexed_data
    )
    for class_id in meta_training_data:
        for index in range(len(meta_training_data[class_id])):
            meta_training_data[class_id][index] = np.transpose(
                meta_training_data[class_id][index],
                (2, 0, 1),
            )

    return meta_training_data
