import torch


def find_span(number_of_control_points: int,
              degree: int,
              evaluation_point: torch.Tensor,
              knot_vector: torch.Tensor
) -> torch.Tensor:
    """
    Determine the knot span index for an evaluation point.

    Later on the basis functions are evaluated, some of them are
    identical to zero and therefore it would be a waste to compute 
    them. That is why first the knot span in which the evaluation 
    point lies is computed using this function.
    See `The NURBS Book` page 68 for reference.

    Parameters
    ----------
    number_of_control_points : int
        The number of control points.
    degree : int 
        The degree of the NURBS surface in a single direction.
    evaluation_point : torch.Tensor
        The evaluation point.
    knot_vector : torch.Tensor
        Contains all the knots of a single direction.
    
    Returns
    -------
    torch.Tensor
        The knot span index.
    """
    # Handle special case: the evaluation point is equal to the last knot
    if evaluation_point == knot_vector[number_of_control_points + 1]:
        return number_of_control_points
    
    # Otherwise perform binary search to find the knot span index
    low = degree
    high = number_of_control_points + 1
    index_middle = (low + high) / 2
    while (evaluation_point < knot_vector[index_middle] or evaluation_point >= knot_vector[index_middle + 1]):
        if (evaluation_point < knot_vector[index_middle]):
            high = index_middle
        else:
            low = index_middle
        index_middle = (low + high) / 2
    return index_middle 