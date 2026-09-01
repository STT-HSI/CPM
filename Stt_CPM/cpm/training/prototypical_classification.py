import torch


def average_support_features(
    support_features: torch.Tensor,
    class_count: int,
    support_samples_per_class: int,
) -> torch.Tensor:
    if support_samples_per_class > 1:
        return support_features.reshape(
            class_count,
            support_samples_per_class,
            -1,
        ).mean(dim=1)
    return support_features


def l2_normalize(
    features: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    return features / (
        features.norm(dim=1, keepdim=True) + eps
    )


def negative_squared_euclidean_logits(
    query_features: torch.Tensor,
    prototypes: torch.Tensor,
) -> torch.Tensor:
    query_count = query_features.shape[0]
    prototype_count = prototypes.shape[0]

    expanded_queries = query_features.unsqueeze(1).expand(
        query_count,
        prototype_count,
        -1,
    )
    expanded_prototypes = prototypes.unsqueeze(0).expand(
        query_count,
        prototype_count,
        -1,
    )
    return -(
        (expanded_queries - expanded_prototypes) ** 2
    ).sum(dim=2)
