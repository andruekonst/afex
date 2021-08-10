from torch import Tensor


def weighted_mse_loss(y_pred: Tensor, y_true: Tensor, weights: Tensor) -> Tensor:
    return (weights * (y_true - y_pred) ** 2).mean()

