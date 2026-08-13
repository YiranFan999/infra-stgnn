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

def mape(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean Absolute Percentage Error."""
    return torch.mean(torch.abs(pred - target) / (torch.abs(target) + 1e-8))

