import torch
import torch.nn as nn
import torch.nn.functional as F


class CategoryRelationalAlignmentLoss(nn.Module):
    """
    Cosine-normalized InfoNCE alignment between category semantics and visual prototypes.
    """

    def __init__(
        self,
        batch_size: int,
        device: torch.device,
        temperature: float = 0.05,
    ) -> None:
        super().__init__()
        self.batch_size = int(batch_size)
        self.register_buffer(
            "temperature",
            torch.tensor(float(temperature), dtype=torch.float32, device=device),
        )
        self.register_buffer(
            "self_mask",
            ~torch.eye(
                self.batch_size * 2,
                self.batch_size * 2,
                dtype=torch.bool,
                device=device,
            ),
        )

    def forward(
        self,
        semantic_features: torch.Tensor,
        visual_prototypes: torch.Tensor,
    ) -> torch.Tensor:
        if semantic_features.shape[0] != visual_prototypes.shape[0]:
            raise ValueError(
                "semantic_features and visual_prototypes must have the same batch size."
            )

        semantic_features = F.normalize(semantic_features, p=2, dim=1)
        visual_prototypes = F.normalize(visual_prototypes, p=2, dim=1)

        batch_size = semantic_features.shape[0]
        representations = torch.cat(
            [semantic_features, visual_prototypes],
            dim=0,
        )
        similarity = representations @ representations.t()

        if batch_size == self.batch_size:
            self_mask = self.self_mask.to(similarity.device)
        else:
            self_mask = ~torch.eye(
                2 * batch_size,
                2 * batch_size,
                dtype=torch.bool,
                device=similarity.device,
            )

        positive_semantic_to_visual = torch.diag(similarity, batch_size)
        positive_visual_to_semantic = torch.diag(similarity, -batch_size)
        positives = torch.cat(
            [positive_semantic_to_visual, positive_visual_to_semantic],
            dim=0,
        )

        temperature = self.temperature.to(similarity.device)
        logits = similarity / temperature
        logits = logits.masked_fill(~self_mask, float("-inf"))
        denominator = torch.logsumexp(logits, dim=1)

        loss = -(positives / temperature - denominator)
        return loss.mean()
