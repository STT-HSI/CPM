from typing import Dict

import numpy as np
import torch
from torch.utils.data import Dataset


class HyperspectralPatchDataset(Dataset):
    """Target-domain patch dataset using the original train/test set encoding."""

    def __init__(self, imdb: Dict[str, np.ndarray], train: bool) -> None:
        self.train = train
        train_indices = np.argwhere(imdb["set"] == 1).flatten()
        test_indices = np.argwhere(imdb["set"] == 3).flatten()
        selected_indices = train_indices if train else test_indices

        data = imdb["data"][:, :, :, selected_indices]
        labels = imdb["Labels"][selected_indices]

        self.patch_data = data.transpose((3, 2, 0, 1))
        self.patch_labels = labels

    def __getitem__(self, index: int):
        return self.patch_data[index], self.patch_labels[index]

    def __len__(self) -> int:
        return len(self.patch_data)
