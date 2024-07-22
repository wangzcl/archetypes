from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import ArrayLike, NDArray


# Not used. Just for reference.
def proj_unit_simplex_1d(x: ArrayLike) -> NDArray[np.float]:
    """
    Project a vector onto the unit simplex.

    Parameters
    ----------
    x : array_like, shape (n,)
        Vector to project.

    Returns
    -------
    x_projected : ndarray, shape (n,)
        Vector projected onto the unit simplex.
    """
    a = 1  # sum of x. 1 for unit simplex.
    x = np.asarray(x)
    n = x.shape[0]
    x_sorted = np.flip(np.sort(x))
    x_cumsum = np.cumsum(x_sorted)
    t = (x_cumsum - a) / np.arange(1, n + 1)
    # t < x_sorted: [T ... T F ... F], the Ts contribute to nonzeros in x_projected.
    index = np.count_nonzero(t < x_sorted) - 1
    tau = t[index]
    x_projected = np.maximum(x - tau, 0)
    return x_projected


# Not used. Just for reference.
def proj_unit_simplex(x: ArrayLike, axis=-1):
    """
    Project each vector along a given axis onto the unit simplex respectively.

    Parameters
    ----------
    x : array_like
        Array to project.

    axis : int, default=-1
        The axis of the input array along which the projection is performed.

    Returns
    -------
    x_projected : ndarray
        Array projected onto the unit simplex.
    """
    a = 1  # sum of x. 1 for unit simplex.
    x = np.asarray(x)
    ndim = x.ndim
    n = x.shape[axis]
    x_sorted = np.flip(np.sort(x, axis=axis), axis=axis)
    x_cumsum = np.cumsum(x_sorted, axis=axis)
    t = (x_cumsum - a) / np.arange(1, n + 1).reshape(
        [1 if i != axis % ndim else n for i in range(ndim)]
    )
    index_array = np.count_nonzero(t < x_sorted, axis=axis, keepdims=True) - 1
    tau = np.take_along_axis(t, index_array, axis=axis)
    x_projected = np.maximum(x - tau, 0)
    return x_projected


class Projector(ABC):
    """
    Abstract base class for projectors.
    """

    @classmethod
    @abstractmethod
    def from_array(cls, x: NDArray, inplace: bool = False) -> "Projector":
        """
        Create a projector from an array to project.

        Parameters
        ----------
        x : ndarray
            Array to project.
        inplace : bool, default=False
            Whether to use the input array as intermediate arrays.

        Returns
        -------
        projector : Projector
            Projector.
        """
        pass

    @abstractmethod
    def project(self, x: NDArray) -> NDArray:
        """
        Project the input array onto the feasible set.

        Parameters
        ----------
        x : ndarray
            Array to project.

        Returns
        -------
        x_projected : ndarray
            Array projected onto the feasible set.
        """
        pass


class UnitSimplexProjector(Projector):
    """
    Direct Projector onto the unit simplex (each row).
    """

    def __init__(
        self,
        shape: tuple[int],
        sorted: NDArray[np.float_],
        cumsum: NDArray[np.float_],
        arange: NDArray[np.int_],
        t_less_xsorted: NDArray[np.bool_],
        index: NDArray[np.intp | np.uintp],
        out: NDArray[np.float_],
    ):

        self.shape = shape

        # Pre-allocated intermediate arrays.
        self.sorted = sorted  # Intermediate array for sorting the array along the axis.
        self.cumsum = cumsum  # Intermediate array for cumulative sum along the axis.
        self.arange = arange  # Intermediate array for the range of indices along the axis.
        self.t_less_xsorted = (
            t_less_xsorted  # Intermediate array for comparison between t and x_sorted.
        )
        self.index = index  # Intermediate array for the index of the threshold tau.
        self.out = out  # Output location for the projected array.

    @classmethod
    def from_shape(cls, shape: tuple[int]):
        """
        Create a `UnitSimplexProjector` from the shape of the array to project.

        Parameters
        ----------
        shape : tuple[int]
            Shape of the array to project.

        Returns
        -------
        projector : UnitSimplexProjector
            Projector onto the unit simplex.
        """
        # ndim = len(shape)
        n = shape[-1]

        sorted = np.empty(shape, dtype=float)
        cumsum = np.empty(shape, dtype=float)
        arange = np.arange(1, n + 1)  # shape: [1 if i != axis % ndim else n for i in range(ndim)]
        t_less_xsorted = np.empty(shape, dtype=bool)
        index = np.empty(
            (shape[0], 1), dtype=int
        )  # shape: [x.shape[i] if i != axis % ndim else 1 for i in range(ndim)]
        out = np.empty(shape, dtype=float)

        return cls(shape, sorted, cumsum, arange, t_less_xsorted, index, out)

    @classmethod
    def from_array(cls, x: NDArray, inplace: bool = False):
        x = np.asarray(x, dtype=float)
        ndim = x.ndim
        shape = x.shape
        n = shape[-1]

        sorted = x if inplace else np.empty_like(x)
        cumsum = np.empty_like(x)
        arange = np.arange(1, n + 1)
        t_less_xsorted = np.empty_like(x, dtype=bool)
        index = np.empty((shape[0], 1), dtype=int)
        out = x if inplace else np.empty_like(x)
        return cls(shape, sorted, cumsum, arange, t_less_xsorted, index, out)

    def project(self, x: NDArray) -> NDArray[np.float_]:
        """
        Project the input onto the unit simplex.

        Parameters
        ----------
        x : ndarray
            Array to project.

        Returns
        -------
        x_projected : ndarray
            Array projected onto the unit simplex.
        """
        axis = -1
        x = np.asarray(x, dtype=float)
        np.copyto(self.sorted, x, casting="no")
        x_sorted = np.flip(self.sorted.sort(axis=axis), axis=axis)
        x_cumsum = np.cumsum(x_sorted, axis=axis, out=self.cumsum)
        t = x_cumsum
        t -= 1  # sum of x. 1 for unit simplex.
        t /= self.arange
        t_less_xsorted = np.less(t, x_sorted, out=self.t_less_xsorted)
        index_array = np.sum(t_less_xsorted, axis=axis, keepdims=True, out=self.index)
        index_array -= 1
        tau = np.take_along_axis(
            t, index_array, axis=axis
        )  # requires memory allocation, no way to avoid it unless using a loop.
        x_projected = np.maximum(np.subtract(x, tau, out=self.sorted), 0, out=self.out)
        return x_projected


class L1NormalizeProjector(Projector):
    """
    first project the vector onto nonnegative subset of the space,
    then normalize the vector onto the unit simplex (dividing by sum).
    """

    def __init__(
        self,
        shape: tuple[int],
        sum: NDArray[np.float_],
        out: NDArray[np.float_],
    ):
        self.shape = shape
        # Pre-allocated intermediate arrays.
        self.sum = sum
        self.out = out

    @classmethod
    def from_shape(cls, shape: tuple[int]):
        """
        Create a `l1NormalizeProjector` from the shape of the array to project.

        Parameters
        ----------
        shape : tuple[int]
            Shape of the array to project.

        Returns
        -------
        projector : l1NormalizeProjector
            Projector onto the unit simplex.
        """
        sum = np.empty((shape[0], 1), dtype=float)
        out = np.empty(shape, dtype=float)
        return cls(shape, sum, out)

    @classmethod
    def from_array(cls, x: NDArray, inplace: bool = False):
        """
        Create a `l1NormalizeProjector` from an array to project.

        Parameters
        ----------
        x : ndarray
            Array to project.
        inplace : bool, default=False
            Whether to use the input array as intermediate arrays.

        Returns
        -------
        projector : l1NormalizeProjector
            Projector onto the unit simplex.
        """
        x = np.asarray(x, dtype=float)
        shape = x.shape
        sum = np.empty((shape[0], 1), dtype=float)
        out = x if inplace else np.empty_like(x)
        return cls(shape, sum, out)

    def project(self, x: np.ndarray) -> np.ndarray[float]:
        x = np.asarray(x, dtype=float)
        x_projected = np.maximum(x, 0, out=self.out)
        sum = np.sum(x_projected, axis=-1, keepdims=True, out=self.sum)
        x_projected /= sum
        return x_projected
