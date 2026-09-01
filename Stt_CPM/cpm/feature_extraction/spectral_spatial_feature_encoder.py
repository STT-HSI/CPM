import torch
import torch.nn as nn

from .spatial_feature_encoder import SpatialFeatureEncoder
from .spectral_feature_encoder import SpectralFeatureEncoder


class SpectralSpatialFeatureEncoder(nn.Module):
    """Shared dual-branch spectral-spatial visual feature encoder."""

    def __init__(
        self,
        shared_spectral_dimension: int,
        patch_size: int,
        embedding_dimension: int,
    ) -> None:
        super().__init__()
        self.spectral_encoder = SpectralFeatureEncoder(
            input_channels=shared_spectral_dimension,
            patch_size=patch_size,
            feature_dimension=embedding_dimension,
        )
        self.spatial_encoder = SpatialFeatureEncoder(
            input_channels=shared_spectral_dimension,
            patch_size=patch_size,
            feature_dimension=embedding_dimension,
        )

    def forward(self, mapped_patch: torch.Tensor) -> torch.Tensor:
        spatial_features = self.spatial_encoder(mapped_patch)
        spectral_features = self.spectral_encoder(mapped_patch)
        return 0.5 * spatial_features + 0.5 * spectral_features
