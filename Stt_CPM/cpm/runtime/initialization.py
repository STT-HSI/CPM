import torch.nn as nn


def initialize_module_weights(module: nn.Module) -> None:
    if isinstance(module, (nn.Conv3d, nn.Conv2d, nn.Conv1d)):
        nn.init.xavier_uniform_(module.weight, gain=1)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.BatchNorm3d, nn.BatchNorm2d, nn.BatchNorm1d)):
        if module.weight is not None:
            nn.init.normal_(module.weight, 1.0, 0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Linear):
        nn.init.xavier_normal_(module.weight)
        if module.bias is not None:
            nn.init.ones_(module.bias)


def initialize_trainable_modules(device, *modules: nn.Module) -> None:
    for module in modules:
        module.apply(initialize_module_weights)
        module.to(device)
