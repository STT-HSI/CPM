from typing import Sequence

import torch


def build_class_identity_vectors(
    class_ids: Sequence[int],
    prior_dimension: int,
    device: torch.device,
) -> torch.Tensor:
    class_identity = torch.zeros(
        len(class_ids),
        prior_dimension,
        device=device,
        dtype=torch.float32,
    )
    for row_index, class_id in enumerate(class_ids):
        class_id = int(class_id)
        if class_id < 0 or class_id >= prior_dimension:
            raise ValueError(
                f"Class id {class_id} is outside label-prior dimension {prior_dimension}."
            )
        class_identity[row_index, class_id] = 1.0
    return class_identity


def generate_label_conditioned_priors(
    class_identity: torch.Tensor,
    noise_std: float,
) -> torch.Tensor:
    perturbation = torch.randn_like(class_identity) * float(noise_std)
    perturbation = perturbation.abs()

    true_class_indices = class_identity.argmax(dim=1, keepdim=True)
    row_maximum = perturbation.max(dim=1, keepdim=True).values
    perturbation.scatter_(
        1,
        true_class_indices,
        row_maximum + float(noise_std),
    )

    label_prior = class_identity + perturbation
    return label_prior / label_prior.sum(dim=1, keepdim=True)
