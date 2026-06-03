import torch


def reduce(tensor: torch.Tensor, reduce: str = "mean", dim: int | None = None) -> torch.Tensor:
    """
    Reduce a tensor based on the specified reduction method.

    Args:
        tensor (torch.Tensor): The input tensor to be reduced.
        reduce (str): The reduction method to apply. Options are 'mean', 'sum', or 'none'.

    Returns:
        torch.Tensor: The reduced tensor.
    """
    if reduce == "mean":
        return torch.mean(tensor, dim=dim)
    if reduce == "sum":
        return torch.sum(tensor, dim=dim)
    if reduce == "none":
        return tensor

    msg = f"Unknown reduction method: {reduce}"
    raise ValueError(msg)
