import torch
import torch.nn as nn


class DomainSpecificSpectralMapping(nn.Module):
    """Map each domain's original spectral dimension into a shared spectral space."""

    def __init__(self, input_dimension: int, output_dimension: int) -> None:
        super().__init__()
        self.projection = nn.Conv2d(
            input_dimension,
            output_dimension,
            kernel_size=1,
            stride=1,
            bias=False,
        )
        self.normalization = nn.BatchNorm2d(output_dimension)

    def forward(self, hyperspectral_patch: torch.Tensor) -> torch.Tensor:
        mapped_patch = self.projection(hyperspectral_patch)
        return self.normalization(mapped_patch)
