from abc import ABC, abstractmethod
from functools import wraps
from typing import Callable
import numpy as np
from numpy.typing import NDArray
from ._projection import Projector, UnitSimplexProjector, L1NormalizeProjector


class PGD(ABC):
    """
    Abstract base class for Projected Gradient Descent Optimizers.

    Attributes
    ----------
    x : ndarray
        Current point.
    y : float
        Value of the objective function at the current point.
    y_new : float
        Value of the objective function at the new point.
    grad : ndarray
        Gradient at the current point.
    x_new : ndarray
        Pre-allocated array for new point after the PGD step.
    projector: Projector
        Projector onto the feasible set.
    """

    @abstractmethod
    def __init__(self, x: NDArray, step_size: float, projector_type: type[Projector], **kwargs):
        self.x = x
        self.x_new = np.empty_like(x)
        self._grad = np.empty_like(x)
        self.step_size = step_size
        self.projector = projector_type.from_array(self.x_new, inplace=True)

    @abstractmethod
    def f(self, x) -> float:
        """The objective function f(x)."""
        pass

    @property
    def y(self):
        """Value of the objective function at the current point."""
        return self.f(self.x)

    @property
    def y_new(self):
        """Value of the objective function at the new point."""
        return self.f(self.x_new)

    def _eval_intermediate_vars(self):
        """Evaluate intermediate variables for grad computing if necessary."""
        pass

    @abstractmethod
    def _eval_grad(self):
        """Update the gradient with current x and other intermediate variables."""
        pass

    @property
    def grad(self):
        return self.grad

    def _descent(self) -> NDArray:
        """Perform a single step of gradient descent."""
        x_new = np.multiply(self.grad, -self.step_size, out=self.x_new)
        x_new += self.x
        return x_new

    def _project(self, x: np.ndarray) -> NDArray:
        """Project the input onto the feasible set."""
        return self.projector.project(x)

    def _pgd_step(self) -> NDArray:
        """Perform a single step of projected gradient descent."""
        x_new = self._project(self._descent())
        return x_new

    def pgd_step(self) -> NDArray:
        """Evaluate current gradient and then perform a single step of projected gradient descent."""
        self._eval_intermediate_vars()
        self._eval_grad()
        x_new = self._pgd_step()
        return x_new

    def update_x(self):
        """Copy the new point `self.x_new` to `self.x` after the PGD step."""
        np.copyto(self.x, self.x_new, casting="no")


class PGD_AA(PGD):
    """Abstract Class for PGD for Archetypal Analysis."""

    @abstractmethod
    def __init__(
        self,
        A: NDArray,
        B: NDArray,
        X: NDArray,
        ABX: NDArray,
        X_ABX: NDArray,
        XXT: NDArray,
        **kwargs,  # kwargs not used, just for compatibility
    ):
        self.A = A
        self.B = B
        self.X = X
        self.XXᵀ = XXT
        self.ABX = ABX
        self.X_ABX = X_ABX
        super().__init__(self, **kwargs)

    @staticmethod
    def auto_vars(A: NDArray, B: NDArray, X: NDArray):
        for var in [A, B, X]:
            assert isinstance(var, np.ndarray) and var.ndim == 2, "Input must be a 2D numpy array."
        assert (
            A.shape[1] == B.shape[0] and B.shape[1] == X.shape[0]
        ), f"Incompatible shapes A: {A.shape}, B:{B.shape}, X: {X.shape} to allow A @ B @ X."
        XXT = np.matmul(X, X.T)
        ABX = np.empty_like(X)
        X_ABX = np.empty_like(X)
        return XXT, ABX, X_ABX

    def f_AB(self) -> float:
        """Objective function f(A, B) = 0.5 * ||X - ABX||^2."""
        # self.ABX evaluated in subclasses' `f`` method
        np.subtract(self.X, self.ABX, out=self.X_ABX)
        y = np.linalg.norm(self.X_ABX) ** 2 * 0.5  # HALF of the Frobenius norm
        return y


class PGD_Optimize_A(PGD_AA):
    "Class for PGD to optimize A in Archetypal Analysis."

    def __init__(self, A, B, X, **kwargs):
        kwargs["x"] = A
        super().__init__(self, A, B, X, **kwargs)
        self.BX = np.empty_like(X, shape=(B.shape[0], X.shape[1]))
        self.BXXᵀBᵀ = np.empty_like(X, shape=(B.shape[0], B.shape[0]))
        self.XXᵀBᵀ = np.empty_like(X, shape=(X.shape[0], B.shape[0]))

    def _eval_intermediate_vars(self):
        self.BX = np.matmul(self.B, self.X, out=self.BX)
        self.BXXᵀBᵀ = np.linalg.multi_dot([self.B, self.XXᵀ, self.B.T], out=self.BXXᵀBᵀ)
        self.XXᵀBᵀ = np.matmul(self.XXᵀ, self.B.T, out=self.XXᵀBᵀ)

    def f(self, A: NDArray) -> float:
        self.ABX = np.matmul(
            A, self.BX, out=self.ABX
        )  # WARNING: BX is not updated until _eval_intermediate_vars or pgd_step is called
        return self.f_AB()

    def _eval_grad(self):
        np.matmul(self.A, self.BXXᵀBᵀ, out=self._grad)
        self._grad -= self.XXᵀBᵀ
        return self._grad


class PGD_Optimize_B(PGD_AA):
    """Class for PGD to optimize `B` in Archetypal Analysis."""

    def __init__(self, A, B, X, **kwargs):
        kwargs["x"] = B
        super().__init__(self, A, B, X, **kwargs)
        self.AᵀA = np.empty_like(A, shape=(A.shape[1], A.shape[1]))
        self.AᵀXXᵀ = np.empty_like(X, shape=(A.shape[1], X.shape[0]))

    def _eval_intermediate_vars(self):
        self.AᵀA = np.matmul(self.A.T, self.A, out=self.AᵀA)
        self.AᵀXXᵀ = np.matmul(self.A.T, self.XXᵀ, out=self.AᵀXXᵀ)

    def f(self, B: NDArray) -> float:
        self.ABX = np.linalg.multi_dot([self.A, B, self.X], out=self.ABX)
        return self.f_AB()

    def _eval_grad(self):
        np.linalg.multi_dot([self.AᵀA, self.B, self.XXᵀ], out=self._grad)
        self._grad -= self.AᵀXXᵀ
        return self._grad


def unit_simplex_proj(cls: type[PGD]):
    """
    A decorator to assert that the PGD uses `UnitSimplexProjector`.
    """
    @wraps(cls.__init__)
    def init_wrapped(self, *args, **kwargs):
        kwargs["projector_type"] = UnitSimplexProjector
        cls.__init__(self, *args, **kwargs)

    cls.__init__ = init_wrapped

    return cls

def l1_norm_proj(cls:type[PGD]):
    """
    A decorator that modifies the gradient computation behavior to fit the L1-normalization "projection".
    That is, g = g - g @ x.
    """
    @wraps(cls.__init__)
    def init_wrapped(self, *args, **kwargs):
        kwargs["projector_type"] = L1NormalizeProjector
        cls.__init__(self, *args, **kwargs)
        self.grad_shift = np.empty_like(self.x, shape=(self.x.shape[-1], 1))

    cls.__init__ = init_wrapped

    @wraps(cls._eval_grad)
    def _eval_grad_wrapped(self):
        cls._eval_grad(self)
        self.grad -= np.linalg.vecdot(self.grad, self.x, out=self.grad_shift)[:, np.newaxis]
        return self.grad

    cls._eval_grad = _eval_grad_wrapped

    return cls

def line_search(cls: type[PGD]):
    """
    Add line search to the PGD class.
    """
    @wraps(cls.__init__)
    def init_wrapped(self, *args, **kwargs):
        cls.__init__(self, *args, **kwargs)
        self.step_size_init = self.step_size
        self.max_search_iter = kwargs.get("max_search_iter", 10)
        self.beta = kwargs.get("beta", 0.5)

    cls.__init__ = init_wrapped

    @wraps(cls._pgd_step)
    def _pgd_step_wrapped(self):
        self.step_size = self.step_size_init
        y_prev = np.inf
        for _ in range(self.max_search_iter):
            x_new = cls._pgd_step(self)
            if self.y_new <= y_prev:
                break
            self.step_size *= self.beta
        return x_new

    cls._pgd_step = _pgd_step_wrapped

    return cls

@line_search
@unit_simplex_proj
class PGD_Optimize_A_UnitSimplex(PGD_Optimize_A):
    pass

@line_search
@unit_simplex_proj
class PGD_Optimize_B_UnitSimplex(PGD_Optimize_B):
    pass

@line_search
@l1_norm_proj
class PGD_Optimize_A_L1Normalize(PGD_Optimize_A):
    pass

@line_search
@l1_norm_proj
class PGD_Optimize_B_L1Normalize(PGD_Optimize_B):
    pass


def gradient_descent_step(x: np.ndarray, grad: np.ndarray, step_size: float) -> np.ndarray:
    """
    Perform a single step of gradient descent.

    Parameters
    ----------
    x : np.ndarray
        Current point.
    grad : np.ndarray
        Gradient at the current point.
    step_size : float
        Step size.

    Returns
    -------
    x_new : np.ndarray
        New point after the gradient descent step.
    """
    x_new = x - step_size * grad
    return x_new


gd_step = gradient_descent_step


def pgd_step(x: np.ndarray, grad: np.ndarray, step_size: float, proj_func: Callable) -> np.ndarray:
    """
    Perform a single step of projected gradient descent.

    Parameters
    ----------
    x : np.ndarray
        Current point.
    grad : np.ndarray
        Gradient at the current point.
    step_size : float
        Step size.
    proj_func : callable
        Projection function.

    Returns
    -------
    x_new : np.ndarray
        New point after the projected gradient descent step.
    """
    x_new = proj_func(gd_step(x, grad, step_size))
    return x_new


def pgd_backtracking_step(
    f: Callable,
    x: np.ndarray,
    grad: np.ndarray,
    step_size: float,
    proj_func: Callable,
    alpha: float = 0.5,
    beta: float = 0.5,
) -> np.ndarray:
    """
    Perform a single step of projected gradient descent with backtracking line search.

    Parameters
    ----------
    f : callable
        Objective function.
    x : np.ndarray
        Current point.
    grad : np.ndarray
        Gradient at the current point.
    step_size : float
        Initial step size.
    proj_func : callable
        Projection function.
    alpha : float, default=0.5
        Factor to decrease the step size.
    beta : float, default=0.5
        Factor to control the backtracking condition.

    Returns
    -------
    x_new : np.ndarray
        New point after the projected gradient descent step with backtracking line search.
    """
    while f(x_new := pgd_step(x, grad, step_size, proj_func)) > f(x) + alpha * ravel_dot(
        grad, (x_new - x)
    ):
        step_size *= beta
    return x_new


def ravel_dot(x: np.ndarray, y: np.ndarray) -> float | int:  # TODO: use vdot!
    """
    Compute the dot product of two raveled arrays.

    Parameters
    ----------
    x : np.ndarray
        First array.
    y : np.ndarray
        Second array.

    Returns
    -------
    dot : float | int
        Dot product of the two raveled arrays.
    """
    dot = np.dot(x.ravel(order="K"), y.ravel(order="K"))
    return dot


def pgd_backtracking_argmin(
    f: Callable,
    nabla_f: Callable,
    x_init: np.ndarray,
    step_size: float,
    proj_func: Callable,
    alpha: float = 0.5,
    beta: float = 0.5,
    max_iter: int = 1000,
    tol: float = 1e-6,
) -> np.ndarray:
    """
    Minimize a function using projected gradient descent with backtracking line search.

    Parameters
    ----------
    f : callable
        Objective function.
    nabla_f : callable
        Gradient of the objective function.
    x_init : np.ndarray
        Initial point.
    step_size : float
        Initial step size.
    proj_func : callable
        Projection function.
    alpha : float, default=0.5
        Factor to decrease the step size.
    beta : float, default=0.5
        Factor to control the backtracking condition.
    max_iter : int, default=1000
        Maximum number of iterations.
    tol : float, default=1e-6
        Tolerance for the stopping condition.

    Returns
    -------
    x_min : np.ndarray
        Point that minimizes the objective function.
    """
    x = x_init
    for _ in range(max_iter):
        grad = nabla_f(x)
        x_new = pgd_backtracking_step(f, x, grad, step_size, proj_func, alpha, beta)
        if (
            abs(f(x_new) - f(x)) < tol
        ):  # TODO: f(x_new) has already been computed in pgd_backtracking_step, can be optimized.
            break
        x = x_new
    return x
