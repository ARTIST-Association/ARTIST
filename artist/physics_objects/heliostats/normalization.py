import typing
import torch
from artist.physics_objects.parameter import AParameter


class ANormalization:
    def normalize(_, parameter: AParameter, value: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    def denormalize(
        _, parameter: AParameter, normalized_value: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError()


class MinMaxNormalization(ANormalization):
    def normalize(_, parameter: AParameter, value: torch.Tensor) -> torch.Tensor:
        if parameter.has_tolerance:
            return 0.5 * (value - parameter.min) / parameter.tolerance
        else:
            return value

    def denormalize(
        _, parameter: AParameter, normalized_value: torch.Tensor
    ) -> torch.Tensor:
        if parameter.has_tolerance:
            return 2 * normalized_value * parameter.tolerance + parameter.min
        else:
            return normalized_value


class ZNormalization(ANormalization):
    def normalize(_, parameter: AParameter, value: torch.Tensor) -> torch.Tensor:
        if parameter.has_tolerance:
            return (value - parameter.initial_value) / parameter.tolerance
        else:
            return value

    def denormalize(
        _, parameter: AParameter, normalized_value: torch.Tensor
    ) -> torch.Tensor:
        if parameter.has_tolerance:
            return normalized_value * parameter.tolerance + parameter.initial_value
        else:
            return normalized_value

class ParameterNormalizer:
    def __init__(
        self,
        normalization_method: typing.Literal[
            "min-max", "z-transformation"
        ] = "z-transformation",
    ):
        self._parameters = {}

        if normalization_method == "z-transformation":
            self._normalization = ZNormalization()
        elif normalization_method == "min-max":
            self._normalization = MinMaxNormalization()
        else:
            raise NotImplementedError()

    def register_parameter(self, parameter: AParameter) -> None:
        self._parameters[parameter.NAME] = parameter

    def get_denormalized_parameter(
        self, name: str, normalized_value: torch.Tensor
    ) -> torch.Tensor:
        try:
            return self._normalization.denormalize(
                self._parameters[name], normalized_value
            )
        except KeyError:
            return normalized_value

    def get_normalized_parameter(self, name: str, value: torch.Tensor) -> torch.Tensor:
        try:
            return self._normalization.normalize(self._parameters[name], value)
        except KeyError:
            return value
