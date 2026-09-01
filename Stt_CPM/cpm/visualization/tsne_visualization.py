from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE


def visualize_features_with_tsne(
    feature_batches: Sequence[torch.Tensor],
    label_batches: Sequence[torch.Tensor],
    class_colors,
    output_path: str,
    random_state: int = 42,
) -> None:
    if not feature_batches:
        raise ValueError("Cannot run t-SNE because no feature batches were collected.")

    features = torch.cat(list(feature_batches), dim=0).numpy()
    labels = torch.cat(list(label_batches), dim=0).numpy()
    sample_count = features.shape[0]

    if sample_count < 2:
        raise ValueError("t-SNE requires at least two samples.")

    perplexity = min(30, max(5, sample_count // 10))
    if perplexity >= sample_count:
        perplexity = max(1, sample_count - 1)

    embedding = TSNE(
        n_components=2,
        random_state=random_state,
        perplexity=perplexity,
    ).fit_transform(features)

    minimum = embedding.min(axis=0)
    maximum = embedding.max(axis=0)
    normalized_embedding = (
        (embedding - minimum)
        / np.maximum(maximum - minimum, 1e-12)
    )

    class_colors = np.asarray(class_colors, dtype=np.float32)
    figure = plt.figure(figsize=(10, 8))
    axes = figure.add_subplot(111)

    for class_id in np.unique(labels):
        class_id_int = int(class_id)
        color_index = min(class_id_int + 1, len(class_colors) - 1)
        class_mask = labels == class_id
        axes.scatter(
            normalized_embedding[class_mask, 0],
            normalized_embedding[class_mask, 1],
            color=[class_colors[color_index]],
            marker="o",
            s=80,
            label=str(class_id_int + 1),
        )

    axes.legend(
        fontsize=15,
        loc="upper right",
        bbox_to_anchor=(1.13, 1.02),
    )
    for spine_name in ["right", "top", "bottom", "left"]:
        axes.spines[spine_name].set_linewidth(2.0)

    axes.tick_params(axis="both", labelsize=20)
    axes.axis("equal")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close(figure)
