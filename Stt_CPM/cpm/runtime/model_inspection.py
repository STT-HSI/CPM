import logging
from typing import Dict

import torch.nn as nn


def _count_trainable_parameters(module: nn.Module) -> int:
    return sum(
        parameter.numel()
        for parameter in module.parameters()
        if parameter.requires_grad
    )


def _count_total_parameters(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def log_model_parameter_counts(
    logger: logging.Logger,
    modules: Dict[str, nn.Module],
) -> None:
    total_trainable = 0
    total_parameters = 0

    logger.info("=" * 72)
    logger.info("Model parameter counts:")
    for name, module in modules.items():
        trainable = _count_trainable_parameters(module)
        total = _count_total_parameters(module)
        total_trainable += trainable
        total_parameters += total
        logger.info(
            "%-32s: %s trainable, %s total",
            name,
            f"{trainable:,}",
            f"{total:,}",
        )

    logger.info("-" * 72)
    logger.info(
        "%-32s: %s trainable, %s total",
        "Total",
        f"{total_trainable:,}",
        f"{total_parameters:,}",
    )
    logger.info("=" * 72)
