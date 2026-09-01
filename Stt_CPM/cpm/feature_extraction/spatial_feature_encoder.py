import torch
import torch.nn as nn


class SpatialFeatureEncoder(nn.Module):
    """3-D convolutional spatial branch used by the shared visual encoder."""

    def __init__(
        self,
        input_channels: int,
        patch_size: int,
        feature_dimension: int,
    ) -> None:
        super().__init__()
        intermediate_channels = 24

        self.spectral_collapse = nn.Conv3d(
            1,
            intermediate_channels,
            kernel_size=(input_channels, 1, 1),
        )
        self.bn1 = nn.BatchNorm3d(intermediate_channels)
        self.relu1 = nn.ReLU()

        self.residual_projection = nn.Conv3d(
            intermediate_channels,
            intermediate_channels,
            kernel_size=(1, 1, 1),
        )

        self.spatial_conv1 = nn.Conv3d(
            intermediate_channels,
            intermediate_channels,
            kernel_size=(1, 3, 3),
            stride=(1, 1, 1),
            padding=(0, 1, 1),
            padding_mode="zeros",
            bias=True,
        )
        self.bn2 = nn.BatchNorm3d(intermediate_channels)
        self.relu2 = nn.ReLU()

        self.spatial_conv2 = nn.Conv3d(
            intermediate_channels,
            intermediate_channels,
            kernel_size=(1, 3, 3),
            stride=(1, 1, 1),
            padding=(0, 1, 1),
            padding_mode="zeros",
            bias=True,
        )
        self.bn3 = nn.BatchNorm3d(intermediate_channels)
        self.relu3 = nn.ReLU()

        self.average_pool = nn.AvgPool3d((1, patch_size, patch_size))
        self.projection = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(intermediate_channels, feature_dimension),
        )

    def forward(self, hyperspectral_patch: torch.Tensor) -> torch.Tensor:
        features = hyperspectral_patch.unsqueeze(1)
        features = self.relu1(self.bn1(self.spectral_collapse(features)))

        residual = self.residual_projection(features)
        features = self.relu2(self.bn2(self.spatial_conv1(features)))
        features = self.spatial_conv2(features)
        features = self.relu3(self.bn3(residual + features))

        features = features.reshape(
            features.size(0),
            features.size(1),
            features.size(3),
            features.size(4),
        )
        features = self.average_pool(features)
        features = features.reshape(features.size(0), -1)
        return self.projection(features)
