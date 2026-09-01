import random
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import Sampler


class EpisodicClassificationTask:
    """Sample an N-way support/query episode from a class-indexed data dictionary."""

    def __init__(
        self,
        data: Dict[int, List[np.ndarray]],
        class_count: int,
        support_samples_per_class: int,
        query_samples_per_class: int,
    ) -> None:
        self.data = data
        self.class_count = class_count
        self.support_samples_per_class = support_samples_per_class
        self.query_samples_per_class = query_samples_per_class

        available_classes = sorted(list(data))
        selected_classes = random.sample(available_classes, class_count)
        local_labels = dict(
            zip(selected_classes, np.arange(len(selected_classes)))
        )

        self.support_data = []
        self.query_data = []
        self.support_labels = []
        self.query_labels = []
        self.support_real_class_ids = []
        self.query_real_class_ids = []

        for class_id in selected_classes:
            sampled_class_data = random.sample(
                self.data[class_id],
                len(self.data[class_id]),
            )
            random.shuffle(sampled_class_data)

            support_end = support_samples_per_class
            query_end = support_end + query_samples_per_class

            self.support_data += sampled_class_data[:support_end]
            self.query_data += sampled_class_data[support_end:query_end]

            self.support_labels += [
                local_labels[class_id]
                for _ in range(support_samples_per_class)
            ]
            self.query_labels += [
                local_labels[class_id]
                for _ in range(query_samples_per_class)
            ]

            self.support_real_class_ids += [
                class_id
                for _ in range(support_samples_per_class)
            ]
            self.query_real_class_ids += [
                class_id
                for _ in range(query_samples_per_class)
            ]


class EpisodicSplitDataset(Dataset):
    def __init__(
        self,
        task: EpisodicClassificationTask,
        split: str = "support",
    ) -> None:
        self.task = task
        self.split = split
        if split == "support":
            self.image_data = task.support_data
            self.labels = task.support_labels
        elif split == "query":
            self.image_data = task.query_data
            self.labels = task.query_labels
        else:
            raise ValueError("split must be 'support' or 'query'.")

    def __len__(self) -> int:
        return len(self.image_data)

    def __getitem__(self, index: int):
        return self.image_data[index], self.labels[index]


class ClassBalancedEpisodeSampler(Sampler):
    def __init__(
        self,
        samples_per_class: int,
        class_count: int,
        available_instances_per_class: int,
        shuffle: bool = True,
    ) -> None:
        self.samples_per_class = samples_per_class
        self.class_count = class_count
        self.available_instances_per_class = available_instances_per_class
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            batch = [
                [
                    i + class_index * self.available_instances_per_class
                    for i in torch.randperm(
                        self.available_instances_per_class
                    )[: self.samples_per_class]
                ]
                for class_index in range(self.class_count)
            ]
        else:
            batch = [
                [
                    i + class_index * self.available_instances_per_class
                    for i in range(self.available_instances_per_class)[
                        : self.samples_per_class
                    ]
                ]
                for class_index in range(self.class_count)
            ]

        flattened_batch = [
            index
            for class_batch in batch
            for index in class_batch
        ]
        if self.shuffle:
            random.shuffle(flattened_batch)
        return iter(flattened_batch)

    def __len__(self) -> int:
        return 1


def build_episode_data_loader(
    task: EpisodicClassificationTask,
    samples_per_class: int,
    split: str,
    shuffle: bool = False,
) -> DataLoader:
    dataset = EpisodicSplitDataset(task, split=split)
    if split == "support":
        available_instances = task.support_samples_per_class
    elif split == "query":
        available_instances = task.query_samples_per_class
    else:
        raise ValueError("split must be 'support' or 'query'.")

    sampler = ClassBalancedEpisodeSampler(
        samples_per_class=samples_per_class,
        class_count=task.class_count,
        available_instances_per_class=available_instances,
        shuffle=shuffle,
    )
    return DataLoader(
        dataset,
        batch_size=samples_per_class * task.class_count,
        sampler=sampler,
    )
