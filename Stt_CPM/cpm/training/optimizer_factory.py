from itertools import chain

import torch

from .model_factory import CPMModelBundle


def build_cpm_optimizers(
    models: CPMModelBundle,
    learning_rate: float,
    weight_decay: float,
):
    return {
        "source_spectral_mapping": torch.optim.Adam(
            models.source_spectral_mapping.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        ),
        "target_spectral_mapping": torch.optim.Adam(
            models.target_spectral_mapping.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        ),
        "visual_and_semantic_projection": torch.optim.Adam(
            chain(
                models.visual_encoder.parameters(),
                models.semantic_projection.parameters(),
            ),
            lr=learning_rate,
            weight_decay=weight_decay,
        ),
        "prototype_refinement": torch.optim.Adam(
            models.prototype_refinement.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        ),
        "semantic_prompt_tuning": torch.optim.Adam(
            models.semantic_aggregation.parameters(),
            lr=learning_rate,
        ),
    }


def zero_optimizer_gradients(optimizers) -> None:
    for optimizer in optimizers.values():
        optimizer.zero_grad()


def step_optimizers(optimizers) -> None:
    for optimizer in optimizers.values():
        optimizer.step()
