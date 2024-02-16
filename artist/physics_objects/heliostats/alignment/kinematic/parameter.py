from typing import Union

import torch


class AParameter:
    name = "<PARAMETER_NAME>"

    def __init__(
        self,
        value: Union[torch.Tensor, float],
        tolerance: Union[torch.Tensor, float] = None,
        distort: bool = False,
        requires_grad: bool = False,
    ):
        self.initial_value = (
            value if isinstance(value, torch.Tensor) else torch.tensor(value)
        )
        self.has_tolerance = tolerance is not None
        self.tolerance = tolerance
        self.requires_grad = requires_grad

        self.min = (
            self.initial_value - self.tolerance if self.has_tolerance else -torch.inf
        )
        self.max = (
            self.initial_value + self.tolerance if self.has_tolerance else torch.inf
        )

        if distort:
            self.distort()

    def distort(self):
        raise NotImplementedError("Must override.")
    
