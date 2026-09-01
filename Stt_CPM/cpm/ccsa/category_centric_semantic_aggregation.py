from typing import Dict, Sequence

import torch
import torch.nn as nn
from peft import PromptTuningConfig, TaskType, get_peft_model
from transformers import AutoModel, AutoTokenizer


def aggregate_category_name_tokens(
    last_hidden_state: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Aggregate non-padding category-name tokens while excluding PEFT virtual prompts.

    PEFT prompt tuning may prepend virtual tokens to the hidden-state sequence without
    extending the tokenizer attention mask. In that case zero-valued mask entries are
    prepended so the virtual prompt positions are excluded from the mean.
    """
    hidden_length = last_hidden_state.size(1)
    mask_length = attention_mask.size(1)

    if hidden_length > mask_length:
        prompt_length = hidden_length - mask_length
        prompt_mask = torch.zeros(
            attention_mask.size(0),
            prompt_length,
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        attention_mask = torch.cat([prompt_mask, attention_mask], dim=1)
    elif hidden_length < mask_length:
        attention_mask = attention_mask[:, :hidden_length]

    expanded_mask = attention_mask.unsqueeze(-1).expand_as(last_hidden_state).float()
    summed_embeddings = torch.sum(last_hidden_state * expanded_mask, dim=1)
    valid_token_count = torch.clamp(expanded_mask.sum(dim=1), min=1e-9)
    return summed_embeddings / valid_token_count


def select_episode_category_semantics(
    category_semantics: torch.Tensor,
    class_ids: Sequence[int],
) -> torch.Tensor:
    indices = torch.tensor(
        class_ids,
        dtype=torch.long,
        device=category_semantics.device,
    )
    if indices.numel() > 0 and int(indices.max()) >= category_semantics.size(0):
        raise IndexError(
            f"Category index {int(indices.max())} exceeds available semantic rows "
            f"{category_semantics.size(0)}."
        )
    return category_semantics.index_select(0, indices)


class CategoryCentricSemanticAggregation(nn.Module):
    """Prompt-tuned category-name encoder with prompt-aware token aggregation."""

    def __init__(
        self,
        model_path: str,
        prompt_virtual_tokens: int,
        token_dimension: int = 768,
        prompt_attention_heads: int = 12,
        prompt_layers: int = 12,
    ) -> None:
        super().__init__()
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.pad_token = self.tokenizer.unk_token

        base_text_model = AutoModel.from_pretrained(model_path)
        prompt_config = PromptTuningConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            num_virtual_tokens=prompt_virtual_tokens,
            token_dim=token_dimension,
            num_attention_heads=prompt_attention_heads,
            num_layers=prompt_layers,
        )
        self.text_model = get_peft_model(base_text_model, prompt_config)

    def tokenize_category_names(
        self,
        category_names: Sequence[str],
    ) -> Dict[str, torch.Tensor]:
        return self.tokenizer(
            list(category_names),
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

    def forward(
        self,
        encoded_category_names: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        outputs = self.text_model(
            input_ids=encoded_category_names["input_ids"],
            attention_mask=encoded_category_names["attention_mask"],
        )
        return aggregate_category_name_tokens(
            outputs.last_hidden_state,
            encoded_category_names["attention_mask"],
        )


class CategorySemanticProjection(nn.Module):
    """Project 768-D MPNet category semantics into the visual embedding space."""

    def __init__(
        self,
        output_dimension: int,
        dropout: float,
        input_dimension: int = 768,
        hidden_dimension: int = 128,
    ) -> None:
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dimension, hidden_dimension, bias=True),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dimension, output_dimension, bias=True),
        )

    def forward(self, category_semantics: torch.Tensor) -> torch.Tensor:
        return self.projection(category_semantics)
