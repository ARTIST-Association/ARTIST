import typing
import torch
from artist.physics_objects.parameter import AParameter


class ANormalization:
    """
    Abstract base class for all normalizers.

    Normalizers implement both normalization and denormalisation functions.
    """

    def normalize(_, parameter: AParameter, value: torch.Tensor) -> torch.Tensor:
        """
        Normalize a parameters value.

        Parameters
        ----------
        parameter : AParameter
            The parameter to be normalized.
        value : torch.Tensor
            The value of the parameter before normalization.

        Raises
        ------
        NotImplementedError
            Whenever called (abstract base class method).
        """
        raise NotImplementedError()

    def denormalize(
        _, parameter: AParameter, normalized_value: torch.Tensor
    ) -> torch.Tensor:
        """
        Denormailze a parameters value.

        Parameters
        ----------
        parameter : AParameter
            The parameter to be denormalized.
        value : torch.Tensor
            The value of the parameter before denormalization.

        Raises
        ------
        NotImplementedError
            Whenever called (abstract base class method).
        """
        raise NotImplementedError()


class MinMaxNormalization(ANormalization):
    """
    This class implements the Min-Max method for normalization.

    See Also
    --------
    :class: ANormailzation: Reference to the parent class
    """

    def normalize(_, parameter: AParameter, value: torch.Tensor) -> torch.Tensor:
        """
        Normalize the parameters value according to the Min-Max normalization.

        Parameters
        ----------
        parameter : AParameter
            The parameter to be normalized.
        value : torch.Tensor
            The value of the parameter before normalization.

        Returns
        -------
        torch.Tensor
            The normalized value.
        """
        if parameter.has_tolerance:
            return 0.5 * (value - parameter.min) / parameter.tolerance
        else:
            return value

    def denormalize(
        _, parameter: AParameter, normalized_value: torch.Tensor
    ) -> torch.Tensor:
        """
        Denormalize the parameters value according to the Min-Max normalization.

        Parameters
        ----------
        parameter : AParameter
            The parameter to be denormalized.
        value : torch.Tensor
            The value of the parameter before denormalization.

        Returns
        -------
        torch.Tensor
            The denormalized value.
        """
        if parameter.has_tolerance:
            return 2 * normalized_value * parameter.tolerance + parameter.min
        else:
            return normalized_value


class ZNormalization(ANormalization):
    """
    This class implements the Z-normalization method for normalization.

    See Also
    --------
    :class: ANormailzation: Reference to the parent class
    """

    def normalize(_, parameter: AParameter, value: torch.Tensor) -> torch.Tensor:
        """
        Normalize the parameters value according to the Z-normalization method.

        Parameters
        ----------
        parameter : AParameter
            The parameter to be normalized.
        value : torch.Tensor
            The value of the parameter before normalization.

        Returns
        -------
        torch.Tensor
            The normalized value.
        """
        if parameter.has_tolerance:
            return (value - parameter.initial_value) / parameter.tolerance
        else:
            return value

    def denormalize(
        _, parameter: AParameter, normalized_value: torch.Tensor
    ) -> torch.Tensor:
        """
        Denormalize the parameters value according to the Z-normalization.

        Parameters
        ----------
        parameter : AParameter
            The parameter to be denormalized.
        value : torch.Tensor
            The value of the parameter before denormalization.

        Returns
        -------
        torch.Tensor
            The denormalized value.
        """
        if parameter.has_tolerance:
            return normalized_value * parameter.tolerance + parameter.initial_value
        else:
            return normalized_value


class ParameterNormalizer:
    """
    This class implemetns a parameter normalizer using either min-max or z-normalization.
    """

    def __init__(
        self,
        normalization_method: typing.Literal[
            "min-max", "z-transformation"
        ] = "z-transformation",
    ) -> None:
        """
        Initialize the parameter normalizer

        Parameters
        ----------
        normalization_method : typing.Literal
            Selects the deired normalization method.
        """
        self._parameters = {}

        if normalization_method == "z-transformation":
            self._normalization = ZNormalization()
        elif normalization_method == "min-max":
            self._normalization = MinMaxNormalization()
        else:
            raise NotImplementedError()

    def register_parameter(self, parameter: AParameter) -> None:
        """
        Add parameter to the module, register it.

        Parameters
        ----------
        parameter : AParameter
            name of the parameter to be registered.
        """
        self._parameters[parameter.name] = parameter

    def get_denormalized_parameter(
        self, name: str, normalized_value: torch.Tensor
    ) -> torch.Tensor:
        """
        Denormalize the passed parameter.

        Parameters
        ----------
        name : str
            The parameter to be denormalized.
        value : torch.Tensor
            The value of the parameter before denormalization.

        Returns
        -------
        torch.Tensor
            The denormalized value.
        """
        try:
            return self._normalization.denormalize(
                self._parameters[name], normalized_value
            )
        except KeyError:
            return normalized_value

    def get_normalized_parameter(self, name: str, value: torch.Tensor) -> torch.Tensor:
        """
        Normalize the passed parameter.

        Parameters
        ----------
        name : str
            The parameter to be normalized.
        value : torch.Tensor
            The value of the parameter before normalization.

        Returns
        -------
        torch.Tensor
            The normalized value.
        """
        try:
            return self._normalization.normalize(self._parameters[name], value)
        except KeyError:
            return value
