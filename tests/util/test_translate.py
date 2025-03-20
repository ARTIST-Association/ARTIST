import pytest
import torch

from artist.util import utils


@pytest.mark.parametrize(
    "east_translation, north_translation, up_translation, expected",
    [
        (
            torch.tensor([1.0]),
            torch.tensor([2.0]),
            torch.tensor([3.0]),
            torch.tensor(
                [[
                    [1.0, 0.0, 0.0, 1.0],
                    [0.0, 1.0, 0.0, 2.0],
                    [0.0, 0.0, 1.0, 3.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]]
            ),
        ),
        (
            torch.tensor([1.0, 0.0, 0.0]),
            torch.tensor([0.0, 1.0, 0.0, 1.0]),
            torch.tensor([0.0, 0.0, 1.0, 1.0]),
            None,
        ),
        (
            torch.tensor([1.0, 0.0, 0.0, 0.0]),
            torch.tensor([0.0, 1.0, 0.0]),
            torch.tensor([0.0, 0.0, 1.0, 1.0]),
            None,
        ),
        (
            torch.tensor([1.0, 0.0, 0.0, 0.0]),
            torch.tensor([0.0, 1.0, 0.0, 0.0]),
            torch.tensor([0.0, 0.0, 1.0]),
            None,
        ),
    ],
)
def test_translate_enu(
    east_translation: torch.Tensor,
    north_translation: torch.Tensor,
    up_translation: torch.Tensor,
    expected: torch.Tensor,
    device: torch.device,
) -> None:
    """
    Test that the correct translation matrix is created.

    Parameters
    ----------
    east_translation : torch.Tensor
        The translation in east direction.
    north_translation : torch.Tensor
        The translation in north direction.
    up_translation : torch.Tensor
        The translation in up direction.
    expected : torch.Tensor
        The expected overall translation or ``None`` if an error is expected.
    device : torch.device
        The device on which to initialize tensors.

    Raises
    ------
    AssertionError
        If test does not complete as expected.
    """
    # Check if the ValueError is thrown as expected.
    if expected is None:
        with pytest.raises(ValueError) as exc_info:
            utils.translate_enu(
                e=east_translation.to(device),
                n=north_translation.to(device),
                u=up_translation.to(device),
                device=device,
            )
        assert (
            "The three tensors containing the east, north, and up translations must have the same shape."
            in str(exc_info.value)
        )
    else:
        # Check if the translation matrix is correct.
        translation_matrix = utils.translate_enu(
            e=east_translation.to(device),
            n=north_translation.to(device),
            u=up_translation.to(device),
            device=device,
        )
        torch.testing.assert_close(
            translation_matrix, expected.to(device), rtol=1e-4, atol=1e-4
        )
