from typing import Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrize as parametrize
from ._coreutils import RowStochastic


class AA(nn.Sequential):
    """
    Perform Archetypal Analysis on a transposed dataset.

    X: (*, n).  B^t: (n, p), A^t: (p, n).
    Each column of B and A is column-stochastic.
    Forward: X @ B^t @ A^t

    WARNING: To perform AA on a dataset X with `n` samples and `m` features,
    it should be transposed to shape (m, n) before passing to this model,
    in order to perform batch training.

    Parameters
    ----------
    n_input: int
        Number of input samples (n).
    n_archetypes: int
        Number of archetypes (p).
    
    """

    def __init__(self, n_input: int, n_archetypes: int, **kwargs):
        super().__init__(
            nn.Linear(n_input, n_archetypes, bias=False, **kwargs),
            nn.Linear(n_archetypes, n_input, bias=False, **kwargs),
        )
        
        for layer in self:
            parametrize.register_parametrization(layer, "weight", RowStochastic())
        
    @property
    def A(self):
        with torch.no_grad():
            return self[1].weight.T
    
    @property
    def B(self):
        with torch.no_grad():
            return self[0].weight.T


def fit(
    model: nn.Module,
    dataloader: Iterable,
    optimizer: torch.optim.Optimizer,
    loss_fn=None,
    epochs: int = 1000,
    reproject_cycle: Optional[int] = None,
    device: Optional[torch.device] = None,
):
    """
    Fit the AA model using the provided dataloader and optimizer.

    Parameters
    ----------
    model: nn.Module
        The AA model to be trained.
    dataloader: Iterable
        An iterable dataloader providing batches of input data.
    optimizer: torch.optim.Optimizer
        The optimizer to use for training.
    loss_fn: callable, optional
        The loss function to use. Defaults to Mean Squared Error.
    epochs: int, default=1000
        Number of epochs to train the model.
    reproject_cycle: int, optional
        If specified, reproject the unconstrained parameters onto the simplex every `reproject_cycle` epochs.
    device: torch.device
        The device to use for training. If None, uses CUDA if available, else CPU.

    Returns
    -------
    model: nn.Module
        The trained AA model.

    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if loss_fn is None:
        loss_fn = nn.MSELoss()

    for i in range(epochs):
        if reproject_cycle is not None and i != 0 and i % reproject_cycle == 0:
            with torch.no_grad():
                for layer in model:
                    weight_orginal = layer.parametrizations.weight.original
                    weight_orginal.copy_(F.normalize(weight_orginal, p=2, dim=0))
            optimizer.state.clear() # Clear optimizer state to avoid issues with momentum, etc.
            
        for batch in dataloader:
            xbatch = batch[0].to(device)

            optimizer.zero_grad()
            loss = loss_fn(model(xbatch), xbatch)
            loss.backward()
            optimizer.step()
                
    return model


def archetypal_analysis(
    X,
    n_archetypes: int,
    n_runs: int = 5,
    verbose: bool = False,
    random_state: int = None,
    epochs: int = 1000,
    batch_size: int = 1,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = torch.float32,
    optimizer_class=torch.optim.AdamW,
    **optimizer_kwargs,
):
    """
    Perform archetypal analysis using PyTorch.

    Parameters
    ----------
    X: arraylike, shape (n, m)
        Input data with `n` samples and `m` features. will be transposed to (m, n) before feeding into the model.
    n_archetypes: int
        Number of archetypes (p).
    n_runs: int, default=5
        Number of runs with different random initializations.
    verbose: bool, default=False
        If True, prints the reconstruction error for each run.
    random_state: int, optional
        Random seed for reproducibility.
    epochs: int, default=1000
        Number of epochs to train the model.
    batch_size: int, default=1
        Batch size for training.
    device: torch.device, optional
        Device to use for training. If None, uses CUDA if available, else CPU.
    dtype: torch.dtype, optional
        Data type for the model parameters. Defaults to torch.float32.
    optimizer_class: torch.optim.Optimizer, default=torch.optim.AdamW
        Optimizer class to use for training.
    optimizer_kwargs: dict
        Additional keyword arguments for the optimizer.

    Returns
    -------
    archetypes: torch.Tensor, shape (p, m)
        The learned archetypes.
    proportions: torch.Tensor, shape (n, p)
        The proportions of each archetype in the samples.
    loss: float
        The reconstruction error of the best run.
    """
    if random_state is not None:
        torch.manual_seed(random_state)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X = torch.as_tensor(X.T, dtype=dtype, device=device)
    assert X.dim() == 2, "Input data X must be 2-dimensional."
    n_input = X.shape[1]
    dataloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X), batch_size=batch_size, shuffle=False
    )

    best_loss = float("inf")
    best_model = None

    for _ in range(n_runs):

        model = AA(n_input, n_archetypes, device=device, dtype=X.dtype)

        optimizer = optimizer_class(model.parameters(), **optimizer_kwargs)

        model = fit(model, dataloader, epochs=epochs, optimizer=optimizer, device=device)

        with torch.no_grad():
            X_hat = model(X)
            loss = F.mse_loss(X_hat, X).item()
            if verbose:
                print(f"Run {_ + 1}/{n_runs}, Reconstruction MSE: {loss:.6f}")
            if loss < best_loss:
                best_loss = loss
                best_model: AA = model

    with torch.no_grad():
        proportions = best_model.A
        archetypes = best_model[0](X).T

    return archetypes, proportions, best_loss
