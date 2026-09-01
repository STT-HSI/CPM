import torch
import torch.nn as nn


class SpectralFeatureEncoder(nn.Module):
    """3-D convolutional spectral branch used by the shared visual encoder."""

    def __init__(
        self,
        input_channels: int,
        patch_size: int,
        feature_dimension: int,
    ) -> None:
        super().__init__()
        intermediate_channels = 24

        self.conv1 = nn.Conv3d(
            1,
            intermediate_channels,
            kernel_size=(7, 1, 1),
            stride=(2, 1, 1),
            padding=(1, 0, 0),
            bias=True,
        )
        self.bn1 = nn.BatchNorm3d(intermediate_channels)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv3d(
            intermediate_channels,
            intermediate_channels,
            kernel_size=(7, 1, 1),
            stride=(1, 1, 1),
            padding=(3, 0, 0),
            padding_mode="zeros",
            bias=True,
        )
        self.bn2 = nn.BatchNorm3d(intermediate_channels)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv3d(
            intermediate_channels,
            intermediate_channels,
            kernel_size=(7, 1, 1),
            stride=(1, 1, 1),
            padding=(3, 0, 0),
            padding_mode="zeros",
            bias=True,
        )
        self.bn3 = nn.BatchNorm3d(intermediate_channels)
        self.relu3 = nn.ReLU()

        spectral_kernel = ((input_channels - 7 + 2) // 2 + 1)
        self.conv4 = nn.Conv3d(
            intermediate_channels,
            feature_dimension,
            kernel_size=(spectral_kernel, 1, 1),
            bias=True,
        )
        self.bn4 = nn.BatchNorm3d(feature_dimension)
        self.relu4 = nn.ReLU()

        self.average_pool = nn.AvgPool3d((1, patch_size, patch_size))

    def forward(self, hyperspectral_patch: torch.Tensor) -> torch.Tensor:
        features = hyperspectral_patch.unsqueeze(1)

        features = self.relu1(self.bn1(self.conv1(features)))

        residual = features
        features = self.relu2(self.bn2(self.conv2(features)))
        features = self.conv3(features)
        features = self.relu3(self.bn3(residual + features))

        features = self.relu4(self.bn4(self.conv4(features)))
        features = features.reshape(
            features.size(0),
            features.size(1),
            features.size(3),
            features.size(4),
        )
        features = self.average_pool(features)
        return features.reshape(features.size(0), -1)
