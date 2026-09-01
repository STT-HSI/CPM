from dataclasses import dataclass

import torch
import torch.nn as nn

from cpm.ccsa import (
    CategoryCentricSemanticAggregation,
    CategorySemanticProjection,
)
from cpm.feature_extraction import (
    DomainSpecificSpectralMapping,
    SpectralSpatialFeatureEncoder,
)
from cpm.prototype_refinement import PrototypeRefinementNetwork


@dataclass
class CPMModelBundle:
    source_spectral_mapping: DomainSpecificSpectralMapping
    target_spectral_mapping: DomainSpecificSpectralMapping
    visual_encoder: SpectralSpatialFeatureEncoder
    semantic_aggregation: CategoryCentricSemanticAggregation
    semantic_projection: CategorySemanticProjection
    prototype_refinement: PrototypeRefinementNetwork

    def train(self) -> None:
        for module in self.modules():
            module.train()

    def eval(self) -> None:
        for module in self.modules():
            module.eval()

    def modules(self):
        return (
            self.source_spectral_mapping,
            self.target_spectral_mapping,
            self.visual_encoder,
            self.semantic_aggregation,
            self.semantic_projection,
            self.prototype_refinement,
        )


def build_cpm_models(
    config,
    device: torch.device,
    model_path: str,
) -> CPMModelBundle:
    source_spectral_mapping = DomainSpecificSpectralMapping(
        input_dimension=config["source_spectral_dimension"],
        output_dimension=config["shared_spectral_dimension"],
    ).to(device)

    target_spectral_mapping = DomainSpecificSpectralMapping(
        input_dimension=config["target_spectral_dimension"],
        output_dimension=config["shared_spectral_dimension"],
    ).to(device)

    visual_encoder = SpectralSpatialFeatureEncoder(
        shared_spectral_dimension=config["shared_spectral_dimension"],
        patch_size=config["patch_size"],
        embedding_dimension=config["embedding_dimension"],
    ).to(device)

    semantic_aggregation = CategoryCentricSemanticAggregation(
        model_path=model_path,
        prompt_virtual_tokens=config["prompt_virtual_tokens"],
        token_dimension=config["text_hidden_dimension"],
        prompt_attention_heads=config["prompt_attention_heads"],
        prompt_layers=config["prompt_layers"],
    ).to(device)

    semantic_projection = CategorySemanticProjection(
        output_dimension=config["embedding_dimension"],
        dropout=config["semantic_projection_dropout"],
        input_dimension=config["text_hidden_dimension"],
        hidden_dimension=128,
    ).to(device)

    prototype_refinement = PrototypeRefinementNetwork(
        visual_embedding_dimension=config["embedding_dimension"],
        label_prior_dimension=config["label_prior_dimension"],
        output_dimension=config["embedding_dimension"],
        dropout=0.5,
    ).to(device)

    return CPMModelBundle(
        source_spectral_mapping=source_spectral_mapping,
        target_spectral_mapping=target_spectral_mapping,
        visual_encoder=visual_encoder,
        semantic_aggregation=semantic_aggregation,
        semantic_projection=semantic_projection,
        prototype_refinement=prototype_refinement,
    )
