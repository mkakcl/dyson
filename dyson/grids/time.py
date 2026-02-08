"""Time grids."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

from typing_extensions import Self

from dyson import numpy as np
from dyson import util
from dyson.grids.grid import BaseGrid
from dyson.representations.enums import Ordering

if TYPE_CHECKING:
    from typing import Any

    from dyson.grids.frequency import ImaginaryFrequencyGrid, RealFrequencyGrid
    from dyson.typing import Array


class BaseTimeGrid(BaseGrid):
    """Base class for time grids."""

    @property
    def domain(self) -> str:
        """Return the domain of the grid."""
        return "time"

    @abstractmethod
    def propagator(  # noqa: D417
        self, energies: Array, chempot: float | Array, **kwargs: Any
    ) -> Array:
        """Get the propagator of a Lehmann representation on the grid.

        Args:
            energies: Energies of the poles.
            chempot: Chemical potential.

        Returns:
            Propagator of the grid.
        """
        pass

    def _lehmann_kernel(
        self,
        energies: Array,
        chempot: float | Array,
        ordering: Ordering = Ordering.ORDERED,
    ) -> Array:
        """Get the kernel of a Lehmann representation on the grid.

        Args:
            energies: Energies of the poles.
            chempot: Chemical potential.
            ordering: Time ordering of the propagator.

        Returns:
            Kernel of a Lehmann representation on the grid.

        Note:
            The kernel is a hook to generalise the resolvent or propagator, depending on whether
            the grid is in the frequency or time domain.
        """
        return self.propagator(energies, chempot, ordering=ordering)


class RealTimeGrid(BaseTimeGrid):
    """Real time grid."""

    eta: float = 1e-2

    _options = {"eta"}

    def __init__(  # noqa: D417
        self, points: Array, weights: Array | None = None, **kwargs: Any
    ) -> None:
        """Initialise the grid.

        Args:
            points: Points of the grid.
            weights: Weights of the grid.
            eta: Broadening factor.
        """
        self._points = np.asarray(points)
        self._weights = np.asarray(weights) if weights is not None else None
        self.set_options(**kwargs)

    @staticmethod
    def _heaviside(points: Array, energies: Array, ordering: Ordering) -> Array:
        """Get the Heaviside term."""
        ordering = Ordering(ordering)
        theta: Array
        if ordering == ordering.ORDERED:
            occ = (energies < 0).astype(np.float64)
            vir = 1.0 - occ
            theta = occ * np.heaviside(points, 0.5) - vir * np.heaviside(-points, 0.5)
        elif ordering == ordering.RETARDED:
            theta = -np.heaviside(-points, 0.5)
        elif ordering == ordering.ADVANCED:
            theta = np.heaviside(points, 0.5)
        else:
            ordering.raise_invalid_representation()
        return theta

    def propagator(  # noqa: D417
        self,
        energies: Array,
        chempot: float | Array,
        ordering: Ordering = Ordering.ORDERED,
    ) -> Array:
        """Get the propagator of a Lehmann representation on the grid.

        Args:
            energies: Energies of the poles.
            chempot: Chemical potential.
            ordering: Time ordering of the resolvent.

        Returns:
            Propagator on the grid.
        """
        ordering = Ordering(ordering)
        grid = np.expand_dims(self.points, axis=tuple(range(1, energies.ndim + 1)))
        energies = np.expand_dims(energies, axis=0)
        if ordering == Ordering.RETARDED:
            phase = np.exp(grid * self.eta)
        elif ordering == Ordering.ADVANCED:
            phase = np.exp(-grid * self.eta)
        elif ordering == Ordering.ORDERED:
            phase = np.exp(-np.abs(grid) * self.eta)
        else:
            ordering.raise_invalid_representation()
        theta = self._heaviside(grid, energies - chempot, ordering)
        propagator = 1.0j * phase * np.exp(1.0j * grid * energies) * theta
        return propagator

    @property
    def reality(self) -> bool:
        """Get the reality of the grid.

        Returns:
            Reality of the grid.
        """
        return True

    @classmethod
    def from_uniform(cls, start: float, stop: float, num: int, eta: float | None = None) -> Self:
        """Create a uniform real time grid.

        Args:
            start: Start of the grid.
            stop: End of the grid.
            num: Number of points in the grid.
            eta: Broadening factor.

        Returns:
            Uniform real time grid.
        """
        points = np.linspace(start, stop, num, endpoint=True)
        return cls(points, eta=eta)

    @classmethod
    def from_dual(cls, other: RealFrequencyGrid) -> Self:
        """Create a grid from another grid in the dual domain (real frequency).

        Args:
            other: Other (real frequency) grid to create from.

        Returns:
            Real time grid.
        """
        if not other.uniformly_spaced:
            raise NotImplementedError("only uniformly spaced grids are supported.")
        if not other.uniformly_weighted:
            raise NotImplementedError("only uniformly weighted grids are supported.")
        spacing = 2.0 * np.pi / (other.separation * len(other))
        num = len(other)
        points = np.linspace(-spacing * num / 2, spacing * num / 2, num, endpoint=False)
        return cls(points, eta=other.eta)


GridRT = RealTimeGrid


class ImaginaryTimeGrid(BaseTimeGrid):
    """Imaginary time grid."""

    def __init__(  # noqa: D417
        self, points: Array, weights: Array | None = None, **kwargs: Any
    ) -> None:
        """Initialise the grid.

        Args:
            points: Points of the grid.
            weights: Weights of the grid.
        """
        self._points = np.asarray(points)
        self._weights = np.asarray(weights) if weights is not None else None
        self.set_options(**kwargs)

    def propagator(  # noqa: D417
        self,
        energies: Array,
        chempot: float | Array,
        ordering: Ordering = Ordering.ORDERED,
    ) -> Array:
        """Get the propagator of a Lehmann representation on the grid.

        Args:
            energies: Energies of the poles.
            chempot: Chemical potential.
            ordering: Time ordering of the resolvent.

        Returns:
            Propagator on the grid.
        """
        grid = np.expand_dims(self.points, axis=tuple(range(1, energies.ndim + 1)))
        energies = np.expand_dims(energies - chempot, axis=0)
        fermi = util.reciprocal(1.0 + util.exp(-self.beta * energies))  # overflow protected
        propagator = -util.exp(-energies * grid) * fermi  # overflow protected
        return propagator.astype(np.complex128)

    @property
    def reality(self) -> bool:
        """Get the reality of the grid.

        Returns:
            Reality of the grid.
        """
        return False

    @property
    def beta(self) -> float:
        """Get the inverse temperature of the grid.

        Returns:
            Inverse temperature of the grid.
        """
        return self.points[-1] - self.points[0]

    @classmethod
    def from_uniform(cls, num: int, beta: float) -> Self:
        """Create a uniform real time grid.

        Args:
            num: Number of points in the grid.
            beta: Inverse temperature.

        Returns:
            Uniform real time grid.
        """
        shift = 0.5 * beta / num
        points = np.linspace(shift, beta + shift, num, endpoint=True)
        return cls(points)

    @classmethod
    def from_dual(cls, other: ImaginaryFrequencyGrid) -> Self:
        """Create a grid from another grid in the dual domain (imaginary frequency).

        Args:
            other: Other (imaginary frequency) grid to create from.

        Returns:
            Imaginary time grid.
        """
        if not other.uniformly_spaced:
            raise NotImplementedError("only uniformly spaced grids are supported.")
        if not other.uniformly_weighted:
            raise NotImplementedError("only uniformly weighted grids are supported.")
        return cls.from_uniform(len(other) * 2, other.beta)  # Use τ:ω ratio of 2:1


GridIT = ImaginaryTimeGrid
