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
        **kwargs: Any,
    ) -> Array:
        """Get the kernel of a Lehmann representation on the grid.

        Args:
            energies: Energies of the poles.
            chempot: Chemical potential.
            kwargs: Additional keyword arguments for the resolvent.

        Returns:
            Kernel of a Lehmann representation on the grid.

        Note:
            The kernel is a hook to generalise the resolvent or propagator, depending on whether
            the grid is in the frequency or time domain.
        """
        return self.propagator(energies, chempot, **kwargs)


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
        **kwargs: Any,
    ) -> Array:
        """Get the propagator of a Lehmann representation on the grid.

        Args:
            energies: Energies of the poles.
            chempot: Chemical potential.
            ordering: Time ordering of the resolvent.

        Returns:
            Propagator on the grid.
        """
        if kwargs:
            raise TypeError(f"propagator() got unexpected keyword argument: {next(iter(kwargs))}")
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
        **kwargs: Any,
    ) -> Array:
        """Get the propagator of a Lehmann representation on the grid.

        Args:
            energies: Energies of the poles.
            chempot: Chemical potential.
            ordering: Time ordering of the resolvent.

        Returns:
            Propagator on the grid.
        """
        if kwargs:
            raise TypeError(f"propagator() got unexpected keyword argument: {next(iter(kwargs))}")
        grid = np.expand_dims(self.points, axis=tuple(range(1, energies.ndim + 1)))
        energies = np.expand_dims(energies, axis=0)
        occ = ((energies - chempot) < 0).astype(np.float64)
        vir = 1.0 - occ
        propagator = np.exp(-energies * (grid - self.beta)) * occ
        propagator -= np.exp(-energies * grid) * vir
        return propagator

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
        tail: Array = 0.0
        for i, moment in enumerate(moments):
            coefficient = (-1) ** i / factorial(i + 1)
            x = self.points ** (i + 1) - self.beta ** i * self.points
            tail -= util.einsum("...,w->w...", moment, coefficient * x)
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
        return self.points[-1] - self.points[0]

    @classmethod
    def from_uniform(cls, num: int, beta: float) -> ImaginaryTimeGrid:
        """Create a uniform real time grid.

        Args:
            num: Number of points in the grid.
            beta: Inverse temperature.

        Returns:
            Uniform real time grid.
        """
        points = np.linspace(0, beta, num, endpoint=True)
        return cls(points)


GridIT = ImaginaryTimeGrid
