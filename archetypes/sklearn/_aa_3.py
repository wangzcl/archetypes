from dataclasses import dataclass

import numpy as np
import optax
from custom_inherit import doc_inherit
from jax import grad
from jax import nn as jnn
from jax import numpy as jnp

from ..utils import nnls
from ._base import AABase


@dataclass
class AAOptimizer:
    A_init: callable
    B_init: callable
    A_optimize: callable
    B_optimize: callable
    fit: callable


@doc_inherit(parent=AABase, style="numpy_with_merge")
class AABase_3(AABase):
    """
    Archetype Analysis.
    """

    def __init__(
        self,
        n_archetypes,
        max_iter=300,
        tol=1e-4,
        init="uniform",
        init_kwargs=None,
        save_init=False,
        verbose=False,
        random_state=None,
        method="nnls",
        method_kwargs=None,
    ):
        super().__init__(
            n_archetypes=n_archetypes,
            max_iter=max_iter,
            tol=tol,
            init=init,
            init_kwargs=init_kwargs,
            save_init=save_init,
            method=method,
            method_kwargs=method_kwargs,
            verbose=verbose,
            random_state=random_state,
        )

    def _init_A(self, X):
        A = np.zeros((X.shape[0], self.n_archetypes), dtype=np.float64)

        ind = self.random_state.choice(self.n_archetypes, X.shape[0], replace=True)

        for i, j in enumerate(ind):
            A[i, j] = 1

        self.A_ = A

    def _init_B(self, X):
        B = np.zeros((self.n_archetypes, X.shape[0]), dtype=np.float64)

        ind = self.init_c_(
            X, self.n_archetypes, random_state=self.random_state, kwargs=self.init_kwargs
        )

        for i, j in enumerate(ind):
            B[i, j] = 1

        self.B_ = B

    def _optim_A(self, X):
        pass

    def _optim_B(self, X):
        pass

    def _compute_archetypes(self, X):
        self.archetypes_ = self.B_ @ X

    def _loss(self, X):
        X_hat = self.A_ @ self.archetypes_
        return np.linalg.norm(X - X_hat) ** 2

    def fit(self, X, y=None, **fit_params):
        # Initialize coefficients
        self._init_A(X)
        self._init_B(X)

        self._compute_archetypes(X)

        if self.save_init:
            self.archetypes_init_ = self.archetypes_.copy()

        rss = self._loss(X)
        self.loss_ = [rss]

        for i in range(self.max_iter):
            # Verbose mode (print RSS)
            if self.verbose and i % 10 == 0:
                print(f"Iteration {i}/{self.max_iter}: RSS = {rss}")

            # Optimize coefficients
            self._optim_A(X)
            self._optim_B(X)

            self._compute_archetypes(X)

            # Compute RSS
            rss = self._loss(X)
            self.loss_.append(rss)
            if abs(self.loss_[-1] - self.loss_[-2]) < self.tol:
                break

        # Set attributes
        self.similarity_degree_ = self.A_
        self.archetypes_similarity_degree_ = self.B_
        self.labels_ = np.argmax(self.A_, axis=1)

        return self

    def transform(self, X):
        return self._optim_A(X)


@doc_inherit(parent=AABase_3, style="numpy_with_merge")
class AA_3(AABase_3):
    """
    Archetype Analysis s.t. |X - ABX|_2^2 is minimized.
    """

    def __init__(
        self,
        n_archetypes,
        max_iter=300,
        tol=1e-4,
        init="uniform",
        init_kwargs=None,
        save_init=False,
        verbose=False,
        random_state=None,
        method="nnls",
        method_kwargs=None,
    ):
        super().__init__(
            n_archetypes=n_archetypes,
            max_iter=max_iter,
            tol=tol,
            init=init,
            init_kwargs=init_kwargs,
            save_init=save_init,
            method=method,
            method_kwargs=method_kwargs,
            verbose=verbose,
            random_state=random_state,
        )

        self._check_parameters_()

    def _check_parameters_(self):
        # Check params for the optimization method
        if self.method == "nnls":
            self.method_c_: AAOptimizer = nnls_optimizer
            self.max_iter_optimizer = self.method_kwargs.get("max_iter_optimizer", None)
            self.const = self.method_kwargs.get("const", 100.0)
        elif self.method == "pgd":
            self.method_c_: AAOptimizer = pgd_optimizer
            self.beta_ = self.method_kwargs.get("beta", 0.5)
            self.n_iter_optimizer = self.method_kwargs.get("n_iter_optimizer", 10)
            self.max_iter_optimizer = self.method_kwargs.get("max_iter_optimizer", 1_000)
            self.learning_rate = self.method_kwargs.get("learning_rate", 1.0)
            self.step_size_A_ = self.learning_rate
            self.step_size_B_ = self.learning_rate
        elif self.method == "jax":
            self.method_c_: AAOptimizer = jax_optimizer
            self.optimizer = self.method_kwargs.get("optimizer", "sgd")
            self.optimizer_kwargs = self.method_kwargs.get(
                "optimizer_kwargs", {"learning_rate": 1e-3}
            )
            if not isinstance(self.optimizer, optax.GradientTransformation):
                self.optimizer = getattr(optax, self.optimizer)

        # TODO: Check if params are valid for the optimization method

    def _init_B(self, X):
        self.method_c_.B_init(self, X)

    def _init_A(self, X):
        self.method_c_.A_init(self, X)

    def _optim_A(self, X):
        return self.method_c_.A_optimize(self, X)

    def _optim_B(self, X):
        return self.method_c_.B_optimize(self, X)

    def fit(self, X, y=None, **fit_params):
        return self.method_c_.fit(self, X, y, **fit_params)


# Non-Negative Least Squares
def _nnls_init_A(self, X):
    super(type(self), self)._init_A(X)


def _nnls_init_B(self, X):
    super(type(self), self)._init_B(X)


def _nnls_optim_B(self, X):
    self.archetypes_ = np.linalg.pinv(self.A_) @ X
    B_ = self.archetypes_
    X_ = X
    self.B_ = nnls(B_, X_, max_iter=self.max_iter_optimizer)


def _nnls_optim_A(self, X):
    B_ = X
    X_ = self.archetypes_
    self.A_ = nnls(B_, X_, max_iter=self.max_iter_optimizer)


def _nnls_fit(self, X, y=None, **fit_params):
    return super(type(self), self).fit(X, y, **fit_params)


nnls_optimizer = AAOptimizer(
    A_init=_nnls_init_A,
    B_init=_nnls_init_B,
    A_optimize=_nnls_optim_A,
    B_optimize=_nnls_optim_B,
    fit=_nnls_fit,
)


# Projected Gradient Descent (Closed Form)
def _pgd_init_A(self, X):
    super(type(self), self)._init_A(X)


def _pgd_init_B(self, X):
    super(type(self), self)._init_B(X)


def _pgd_optim_A(self, X):
    C_ = self.B_.T
    S_ = self.A_.T
    GS_ = self.GA_.T

    # Pre-computations
    CTXTXC = C_.T @ self.XTX_ @ C_
    CTXTX = C_.T @ self.XTX_
    SST = S_ @ S_.T
    rss_prev = -2 * np.sum(CTXTX * S_) + np.sum(CTXTXC * SST)

    for _ in range(self.n_iter_optimizer):
        # Compute the gradient
        GS_ = CTXTXC @ S_ - CTXTX
        GS_ = GS_ - np.sum(GS_ * S_, axis=0)

        # Optimize the step size
        S_prev = S_.copy()
        for _ in range(self.max_iter_optimizer):
            S_ = S_prev - self.step_size_A_ * GS_
            S_ = np.where(S_ < 0, 1e-8, S_)
            S_ = S_ / np.sum(S_, axis=0)

            SST = S_ @ S_.T
            rss = -2 * np.sum(CTXTX * S_) + np.sum(CTXTXC * SST)

            if rss <= rss_prev:
                self.step_size_A_ /= self.beta_
                rss_prev = rss
                break

            self.step_size_A_ *= self.beta_

    self.GA_ = GS_.T
    self.A_ = S_.T


def _pgd_optim_B(self, X):
    C_ = self.B_.T
    S_ = self.A_.T
    GC_ = self.GB_.T

    # Pre-computations
    SST = S_ @ S_.T
    XTXST = self.XTX_ @ S_.T

    rss_prev = -2 * np.sum(XTXST * C_) + np.sum(C_.T @ self.XTX_ @ C_ * SST)

    for _ in range(self.n_iter_optimizer):
        # Compute the gradient
        GC_ = self.XTX_ @ C_ @ SST - XTXST
        GC_ = GC_ - np.sum(GC_ * C_, axis=0)

        # Optimize the step size
        C_prev = C_.copy()
        for _ in range(self.max_iter_optimizer):
            C_ = C_prev - self.step_size_B_ * GC_
            C_ = np.where(C_ < 1e-8, 0, C_)
            C_ = C_ / np.sum(C_, axis=0)

            rss = -2 * np.sum(XTXST * C_) + np.sum(C_.T @ self.XTX_ @ C_ * SST)
            if rss <= rss_prev:
                self.step_size_B_ /= self.beta_
                rss_prev = rss
                break

            self.step_size_B_ *= self.beta_

    self.GB_ = GC_.T
    self.B_ = C_.T


def _pgd_fit(self, X, y=None, **fit_params):
    # Pre-computations for optimization
    self.GA_ = np.zeros((X.shape[0], self.n_archetypes))
    self.GB_ = np.zeros((self.n_archetypes, X.shape[0]))
    self.XTX_ = X @ X.T

    return super(type(self), self).fit(X, y, **fit_params)


pgd_optimizer = AAOptimizer(
    A_init=_pgd_init_A,
    B_init=_pgd_init_B,
    A_optimize=_pgd_optim_A,
    B_optimize=_pgd_optim_B,
    fit=_pgd_fit,
)


# Gradient Descent (JAX)
def _jax_init_A(self, X):
    super(type(self), self)._init_A(X)
    self.A_opt_ = jnp.asarray(self.A_, copy=True)
    self.A_ = self.A_opt_
    self.optimizer_A_state = self.optimizer_A.init(self.A_opt_)


def _jax_init_B(self, X):
    super(type(self), self)._init_B(X)
    self.B_opt_ = jnp.asarray(self.B_, copy=True)
    self.B_ = self.B_opt_
    self.optimizer_B_state = self.optimizer_B.init(self.B_opt_)


def jax_optim_A(self, X):
    grad_A = grad(_jax_loss, argnums=1)(X, self.A_, self.B_)
    updates_A, self.optimizer_A_state = self.optimizer_A.update(grad_A, self.optimizer_A_state)
    self.A_opt_ = optax.apply_updates(self.A_opt_, updates_A)
    self.A_ = jnn.softmax(self.A_opt_, axis=1)


def jax_optim_B(self, X):
    grad_B = grad(_jax_loss, argnums=2)(X, self.A_, self.B_)
    updates_B, self.optimizer_B_state = self.optimizer_B.update(grad_B, self.optimizer_B_state)
    self.B_opt_ = optax.apply_updates(self.B_opt_, updates_B)
    self.B_ = jnn.softmax(self.B_opt_, axis=1)


def _jax_fit(self, X, y=None, **fit_params):
    # Pre-computations for optimization
    self.optimizer_A = self.optimizer(**self.optimizer_kwargs)
    self.optimizer_B = self.optimizer(**self.optimizer_kwargs)

    return super(type(self), self).fit(X, y, **fit_params)


def _jax_loss(X, A, B):
    return optax.l2_loss(X - A @ B @ X).sum()


jax_optimizer = AAOptimizer(
    A_init=_jax_init_A,
    B_init=_jax_init_B,
    A_optimize=jax_optim_A,
    B_optimize=jax_optim_B,
    fit=_jax_fit,
)
