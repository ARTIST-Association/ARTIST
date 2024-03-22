import torch


class AModule(torch.nn.Module):
    """
    This is an abstract base class for all modules.
    """

    def __init__(self):
        super().__init__()

