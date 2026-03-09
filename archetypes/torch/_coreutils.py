import torch
import torch.nn as nn
import torch.nn.functional as F


class RowStochastic(nn.Module):
    """
    A row-stochastic parameterization for a matrix.
    Enforce each row to lie on the unit simplex, i.e., non-negative and sum to 1.
    """

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return F.normalize(X**2, p=1, dim=1)
