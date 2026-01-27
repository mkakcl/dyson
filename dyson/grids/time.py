"""Time grids."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING
from math import factorial

from dyson import numpy as np
from dyson import util
from dyson.grids.grid import BaseGrid
from dyson.representations.enums import Component, Ordering, Reduction

if TYPE_CHECKING:
    from typing import Any, Iterable

    from dyson.representations.dynamic import Dynamic
    from dyson.representations.lehmann import Lehmann
    from dyson.typing import Array
    from dyson.grids.frequency import GridRF, GridIF


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

    @staticmethod
    def _heaviside(points: Array, energies: Array, ordering: Ordering) -> Array:
        """Get the Heaviside term with complex phase."""
        ordering = Ordering(ordering)
        theta: Array
        if ordering == ordering.ORDERED:
            pos = -1.0j * (points > 0).astype(np.float64) + 0.5 * (points == 0).astype(np.float64)
            neg = 1.0j * (points < 0).astype(np.float64) + 0.5 * (points == 0).astype(np.float64)
            occ = (energies < 0).astype(np.float64)
            vir = 1.0 - occ
            theta = pos * occ + neg * vir
        elif ordering == ordering.ADVANCED:
            pos = -1.0j * (points > 0).astype(np.float64) + 0.5 * (points == 0).astype(np.float64)
            theta = pos
        elif ordering == ordering.RETARDED:
            neg = 1.0j * (points < 0).astype(np.float64) + 0.5 * (points == 0).astype(np.float64)
            theta = neg
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
        grid = np.expand_dims(self.points, axis=tuple(range(1, energies.ndim + 1)))
        energies = np.expand_dims(energies, axis=0)
        phase = np.exp(-1.0j * grid * energies)
        theta = self._heaviside(grid, energies - chempot, ordering)
        return phase * theta

    def evaluate_tail(
        self,
        moments: Iterable[Array],
        ordering: Ordering = Ordering.ORDERED,
    ) -> Array:
        """Evaluate the tail (short time) on the grid, via a moment expansion.

        Args:
            moments: Moments of the tail expansion.

        Returns:
            Values of the tail expansion on the grid.
        """
        raise NotImplementedError

    @property
    def reality(self) -> bool:
        """Get the reality of the grid.

        Returns:
            Reality of the grid.
        """
        return True

    @classmethod
    def from_uniform(cls, start: float, stop: float, num: int) -> RealTimeGrid:
        """Create a uniform real time grid.

        Args:
            start: Start of the grid.
            stop: End of the grid.
            num: Number of points in the grid.

        Returns:
            Uniform real time grid.
        """
        points = np.linspace(start, stop, num, endpoint=True)
        return cls(points)

    #TODO: implement for all classes
    @classmethod
    def from_inverse(cls, grid_rf: GridRF) -> RealTimeGrid:
        """Create a real time grid from the inverse of a real frequency grid.

        Args:
            grid_rf: Real frequency grid.

        Returns:
            Real time grid.
        """
        if not grid_rf.uniformly_spaced:
            raise NotImplementedError("only uniformly spaced grids are supported.")
        spacing = 2.0 * np.pi / (grid_rf.separation * len(grid_rf))
        num = len(grid_rf)
        points = np.linspace(-spacing * num / 2, spacing * num / 2, num, endpoint=False)
        return cls(points)


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
        fermi = 1.0 / (1.0 + np.exp(self.beta * energies))
        propagator = np.exp(-energies * grid) * fermi
        return propagator.astype(np.complex128)

    def evaluate_tail(
        self,
        moments: Iterable[Array],
        ordering: Ordering = Ordering.ORDERED,
    ) -> Array:
        """Evaluate the tail (short time) on the grid, via a moment expansion.

        Args:
            moments: Moments of the tail expansion.

        Returns:
            Values of the tail expansion on the grid.
        """
        orders = [
            lambda x: -1/2 * np.sign(x),
            lambda x: -1/4 * (self.beta - 2.0 * np.abs(x)),
            lambda x: +1/4 * x * (self.beta - np.abs(x)),
            lambda x: +1/48 * (self.beta**3 - 6.0 * self.beta * x ** 2 + 4.0 * np.abs(x ** 3)),
            lambda x: -1/48 * x * (self.beta**3 - 2.0 * self.beta * x ** 2 + np.abs(x ** 3)),
        ]
        tail: Array = 0.0
        for i, moment in enumerate(moments):
            if i >= len(orders):
                raise NotImplementedError(
                    f"{self.__class__.__name__}.evaluate_tail only supports up to order "
                    f"{len(orders)-1}."
                )
            tail -= util.einsum("...,w->w...", moment, orders[i](self.points))
        return tail

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
        return -(self.points[0] + self.points[-1])

    @classmethod
    def from_uniform(cls, num: int, beta: float) -> ImaginaryTimeGrid:
        """Create a uniform real time grid.

        Args:
            num: Number of points in the grid.
            beta: Inverse temperature.

        Returns:
            Uniform real time grid.
        """
        spacing = beta / num
        points = np.linspace(-beta + spacing * 0.5, -spacing * 0.5, num, endpoint=True)
        return cls(points)


GridIT = ImaginaryTimeGrid
