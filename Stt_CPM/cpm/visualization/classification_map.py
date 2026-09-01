from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def save_classification_map_figure(
    rgb_map: np.ndarray,
    ground_truth: np.ndarray,
    dpi: int,
    output_path: str,
) -> None:
    figure = plt.figure(frameon=False)
    figure.set_size_inches(
        ground_truth.shape[1] * 2.0 / dpi,
        ground_truth.shape[0] * 2.0 / dpi,
    )
    axes = plt.Axes(figure, [0.0, 0.0, 1.0, 1.0])
    axes.set_axis_off()
    axes.xaxis.set_visible(False)
    axes.yaxis.set_visible(False)
    figure.add_axes(axes)
    axes.imshow(rgb_map)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi)
    plt.close(figure)


def save_classification_map(
    predictions,
    ground_truth,
    permutation,
    row_indices,
    column_indices,
    train_count: int,
    patch_size: int,
    class_colors,
    output_path: str,
) -> None:
    if ground_truth is None or permutation is None or len(predictions) == 0:
        return

    rendered_ground_truth = ground_truth.copy()
    for prediction_index in range(len(predictions)):
        source_index = permutation[train_count + prediction_index]
        rendered_ground_truth[
            row_indices[source_index],
            column_indices[source_index],
        ] = predictions[prediction_index] + 1

    class_colors = np.asarray(class_colors, dtype=np.float32)
    ground_truth_indices = rendered_ground_truth.astype(np.int64)
    ground_truth_indices = np.clip(
        ground_truth_indices,
        0,
        len(class_colors) - 1,
    )
    rgb_map = class_colors[ground_truth_indices]

    half_width = patch_size // 2
    save_classification_map_figure(
        rgb_map[
            half_width:-half_width,
            half_width:-half_width,
            :,
        ],
        rendered_ground_truth[
            half_width:-half_width,
            half_width:-half_width,
        ],
        dpi=24,
        output_path=output_path,
    )
