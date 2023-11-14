import torch


class AModule(torch.nn.Module):
    def __init__(self):
        super(AModule, self).__init__()
        # self._optimizer_cfg = OptimizerConfig()

    def get_optimizer_config(self):
        return self._optimizer_cfg

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Must Be Overridden!")
