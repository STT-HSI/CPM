import torch
import torch.nn as nn
import torch.nn.functional as F


class PrototypeRefinementNetwork(nn.Module):
    """Refine a visual prototype conditioned on its label prior."""

    def __init__(
        self,
        visual_embedding_dimension: int,
        label_prior_dimension: int,
        output_dimension: int = 128,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        input_dimension = visual_embedding_dimension + label_prior_dimension
        self.projection = nn.Linear(input_dimension, output_dimension)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, conditioned_prototype: torch.Tensor) -> torch.Tensor:
        conditioned_prototype = conditioned_prototype.view(
            conditioned_prototype.size(0),
            -1,
        )
        refined_prototype = F.relu(self.projection(conditioned_prototype))
        return self.dropout(refined_prototype)
