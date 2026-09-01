import logging
from typing import Dict

import numpy as np
import torch
from sklearn.neighbors import KNeighborsClassifier
from torch.utils.tensorboard import SummaryWriter

from cpm.ccsa import select_episode_category_semantics
from cpm.training.model_factory import CPMModelBundle
from cpm.visualization import visualize_features_with_tsne


def _set_target_evaluation_mode(models: CPMModelBundle) -> None:
    models.target_spectral_mapping.eval()
    models.visual_encoder.eval()
    models.semantic_aggregation.eval()
    models.semantic_projection.eval()


def _restore_target_training_mode(models: CPMModelBundle) -> None:
    models.target_spectral_mapping.train()
    models.visual_encoder.train()
    models.semantic_aggregation.train()
    models.semantic_projection.train()


def evaluate_target_domain(
    models: CPMModelBundle,
    train_loader,
    test_loader,
    encoded_target_category_names: Dict[str, torch.Tensor],
    device: torch.device,
    writer: SummaryWriter,
    episode_index: int,
    logger: logging.Logger,
    enable_tsne: bool,
    tsne_output_path: str,
    class_colors,
):
    _set_target_evaluation_mode(models)

    total_correct = 0
    predictions = np.array([], dtype=np.int64)
    labels = np.array([], dtype=np.int64)
    tsne_feature_batches = []
    tsne_label_batches = []

    target_category_semantics = models.semantic_aggregation(
        encoded_target_category_names
    )

    train_patches, train_labels = next(iter(train_loader))
    train_label_indices = train_labels.long().to(device)
    semantic_support = target_category_semantics.index_select(
        0,
        train_label_indices,
    )
    _ = models.semantic_projection(semantic_support)

    train_features = models.visual_encoder(
        models.target_spectral_mapping(train_patches.to(device))
    )

    maximum = train_features.max()
    minimum = train_features.min()
    denominator = (maximum - minimum).clamp(min=1e-12)
    normalized_train_features = (
        train_features - minimum
    ) / denominator

    classifier = KNeighborsClassifier(n_neighbors=1)
    classifier.fit(
        normalized_train_features.cpu().detach().numpy(),
        train_labels.numpy(),
    )

    for test_patches, test_labels in test_loader:
        test_features = models.visual_encoder(
            models.target_spectral_mapping(test_patches.to(device))
        )
        normalized_test_features = (
            test_features - minimum
        ) / denominator

        test_features_cpu = normalized_test_features.detach().cpu()
        test_labels_cpu = test_labels.detach().cpu()

        if enable_tsne:
            tsne_feature_batches.append(test_features_cpu)
            tsne_label_batches.append(test_labels_cpu)

        predicted_labels = classifier.predict(
            test_features_cpu.numpy()
        )
        test_labels_numpy = test_labels_cpu.numpy()

        total_correct += np.sum(
            predicted_labels == test_labels_numpy
        )
        predictions = np.append(predictions, predicted_labels)
        labels = np.append(labels, test_labels_numpy)

    test_accuracy = (
        100.0 * total_correct / len(test_loader.dataset)
    )
    writer.add_scalar(
        "Accuracy/target_test",
        test_accuracy,
        episode_index + 1,
    )

    logger.info(
        "Target test accuracy: %s/%s (%.2f%%)",
        total_correct,
        len(test_loader.dataset),
        test_accuracy,
    )

    if enable_tsne:
        visualize_features_with_tsne(
            feature_batches=tsne_feature_batches,
            label_batches=tsne_label_batches,
            class_colors=class_colors,
            output_path=tsne_output_path,
        )

    _restore_target_training_mode(models)
    return test_accuracy, labels, predictions
