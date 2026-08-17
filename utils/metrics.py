"""
Provide four metrics for forecasting tasks:
Mean Absolute Error (MAE),
Mean Squared Error (MSE),
Root Mean Squared Error (RMSE),
and Mean Absolute Percentage Error (MAPE).
"""

import torch

def mae(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean Absolute Error."""
    return torch.mean(torch.abs(pred - target))

def mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean Squared Error."""
    return torch.mean((pred - target) ** 2)

def rmse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Root Mean Squared Error."""
    return torch.sqrt(torch.mean((pred - target) ** 2))

def mape(pred: torch.Tensor, target: torch.Tensor, threshold: float = 1.0) -> torch.Tensor:
    """Mean Absolute Percentage Error, ignoring targets below threshold to avoid division by near-zero."""
    mask = torch.abs(target) > threshold
    if mask.sum() == 0:
        return torch.tensor(0.0, device=pred.device)
    return torch.mean(torch.abs(pred[mask] - target[mask]) / torch.abs(target[mask]))

