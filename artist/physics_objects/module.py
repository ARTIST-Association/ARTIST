import torch


class AModule(torch.nn.Module):
    """
    [INSERT DESCRIPTION HERE!].

    Methods
    -------
    get_optimizer()
        [INSERT DESCRIPTION HERE!]
    forward()
        [INSERT DESCRIPTION HERE!]
    """

    def __init__(self):
        super().__init__()

    def get_optimizer_config(self):
        """
        [INSERT DESCRIPTION HERE!].

        Returns
        -------
        [INSERT DESCRIPTION HERE!]
        """
        return self._optimizer_cfg

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        [INSERT DESCRIPTION HERE!].

        Parameters
        ----------
        inputs : torch.Tensor
            [INSERT DESCRIPTION HERE!]

        Returns
        -------
        torch.Tensor
            [INSERT DESCRIPTION HERE!]
        """
        raise NotImplementedError("Must be overridden!")
