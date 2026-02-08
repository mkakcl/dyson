"""Frequency grids."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

import scipy.special
from typing_extensions import Self

from dyson import numpy as np
from dyson.grids.grid import BaseGrid, BaseImaginaryGrid, BaseRealGrid
from dyson.representations.enums import Ordering

if TYPE_CHECKING:
    from typing import Any

    from dyson.grids.time import ImaginaryTimeGrid, RealTimeGrid
    from dyson.typing import Array


class BaseFrequencyGrid(BaseGrid):
    """Base class for frequency grids."""

    @property
    def domain(self) -> str:
        """Get the domain of the grid.

        Returns:
            Domain of the grid.
        """
        return "frequency"

    @abstractmethod
    def resolvent(  # noqa: D417
        self,
        energies: Array,
        chempot: float | Array,
        ordering: Ordering = Ordering.ORDERED,
        invert: bool = True,
    ) -> Array:
        """Get the resolvent of a Lehmann representation on the grid.

        Args:
            energies: Energies of the poles.
            chempot: Chemical potential.
            ordering: Time ordering of the resolvent.
            invert: Whether to apply the inversion in the resolvent formula.

        Returns:
            Resolvent on the grid.
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
            ordering: Time ordering of the resolvent.

        Returns:
            Kernel of a Lehmann representation on the grid.

        Note:
            The kernel is a hook to generalise the resolvent or propagator, depending on whether
            the grid is in the frequency or time domain.
        """
        return self.resolvent(energies, chempot, ordering=ordering)


class RealFrequencyGrid(BaseFrequencyGrid, BaseRealGrid):
    """Real frequency grid."""

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
    def _resolvent_signs(energies: Array, ordering: Ordering) -> Array:
        """Get the signs for the resolvent based on the time ordering."""
        ordering = Ordering(ordering)
        signs: Array
        if ordering == ordering.ORDERED:
            signs = np.where(energies >= 0, 1.0, -1.0)
        elif ordering == ordering.ADVANCED:
            signs = -np.ones_like(energies)
        elif ordering == ordering.RETARDED:
            signs = np.ones_like(energies)
        else:
            ordering.raise_invalid_representation()
        return signs

    def resolvent(  # noqa: D417
        self,
        energies: Array,
        chempot: float | Array,
        ordering: Ordering = Ordering.ORDERED,
        invert: bool = True,
    ) -> Array:
        r"""Get the resolvent of a Lehmann representation on the grid.

        For real frequency grids, the resolvent is given by

        .. math::
            R(\omega) = \frac{1}{\omega - E \pm i \eta},

        where :math:`\eta` is a small broadening factor, and :math:`E` are the pole energies. The
        sign of :math:`i \eta` depends on the time ordering of the resolvent.

        Args:
            energies: Energies of the poles.
            chempot: Chemical potential.
            ordering: Time ordering of the resolvent.
            invert: Whether to apply the inversion in the resolvent formula.

        Returns:
            Resolvent on the grid.
        """
        signs = self._resolvent_signs(energies - chempot, ordering)
        grid = np.expand_dims(self.points, axis=tuple(range(1, energies.ndim + 1)))
        energies = np.expand_dims(energies, axis=0)
        denominator = grid + (signs * 1.0j * self.eta - energies)
        return 1.0 / denominator if invert else denominator

    @classmethod
    def from_uniform(cls, start: float, stop: float, num: int, eta: float | None = None) -> Self:
        """Create a uniform real frequency grid.

        Args:
            start: Start of the grid.
            stop: End of the grid.
            num: Number of points in the grid.
            eta: Broadening factor.

        Returns:
            Uniform real frequency grid.
        """
        points = np.linspace(start, stop, num, endpoint=True)
        return cls(points, eta=eta)

    @classmethod
    def from_dual(cls, other: RealTimeGrid) -> Self:
        """Create a grid from another grid in the dual domain (real time).

        Args:
            other: Other (real time) grid to create from.

        Returns:
            Real frequency grid.
        """
        if not other.uniformly_spaced:
            raise NotImplementedError("only uniformly spaced grids are supported.")
        if not other.uniformly_weighted:
            raise NotImplementedError("only uniformly weighted grids are supported.")
        if other.domain != "time" or not other.reality:
            raise ValueError(f"dual grid for {cls.__name__} must be of type RealTimeGrid")
        spacing = 2.0 * np.pi / (other.separation * len(other))
        num = len(other)
        points = np.linspace(-spacing * (num - 1) / 2, spacing * (num + 1) / 2, num, endpoint=False)
        return cls(points, eta=other.eta)


GridRF = RealFrequencyGrid


class ImaginaryFrequencyGrid(BaseFrequencyGrid, BaseImaginaryGrid):
    """Imaginary frequency grid."""

    def __init__(  # noqa: D417
        self, points: Array, weights: Array | None = None, **kwargs: Any
    ) -> None:
        """Initialise the grid.

        Args:
            points: Points of the grid.
            weights: Weights of the grid.
            beta: Inverse temperature.
        """
        self._points = np.asarray(points)
        self._weights = np.asarray(weights) if weights is not None else None
        self.set_options(**kwargs)

    def resolvent(  # noqa: D417
        self,
        energies: Array,
        chempot: float | Array,
        ordering: Ordering = Ordering.ORDERED,
        invert: bool = True,
    ) -> Array:
        r"""Get the resolvent of a Lehmann representation on the grid.

        For imaginary frequency grids, the resolvent is given by

        .. math::
            R(i \omega_n) = \frac{1}{i \omega_n - E},

        where :math:`E` are the pole energies.

        Args:
            energies: Energies of the poles.
            chempot: Chemical potential.
            ordering: Time ordering of the resolvent.
            invert: Whether to apply the inversion in the resolvent formula.

        Returns:
            Resolvent on the grid.
        """
        grid = np.expand_dims(self.points, axis=tuple(range(1, energies.ndim + 1)))
        energies = np.expand_dims(energies, axis=0)
        denominator = 1.0j * grid - energies
        return 1.0 / denominator if invert else denominator

    @classmethod
    def from_uniform(cls, num: int, beta: float | None = None) -> Self:
        """Create a uniform imaginary frequency grid.

        Args:
            num: Number of points in the grid.
            beta: Inverse temperature.

        Returns:
            Uniform imaginary frequency grid.
        """
        if beta is None:
            beta = cls.beta
        points = (2 * np.arange(num) + 1) * np.pi / beta
        return cls(points, beta=beta)

    @classmethod
    def from_legendre(
        cls, num: int, diffuse_factor: float = 1.0, beta: float | None = None
    ) -> Self:
        """Create a Legendre imaginary frequency grid.

        Args:
            num: Number of points in the grid.
            diffuse_factor: Diffuse factor for the grid.
            beta: Inverse temperature.

        Returns:
            Legendre imaginary frequency grid.
        """
        points, weights = scipy.special.roots_legendre(num)
        points = (1 - points) / (diffuse_factor * (1 + points))
        return cls(points, weights=weights, beta=beta)

    @classmethod
    def from_dual(cls, other: ImaginaryTimeGrid) -> Self:
        """Create a grid from another grid in the dual domain (imaginary time).

        Args:
            other: Other (imaginary time) grid to create from.

        Returns:
            Imaginary frequency grid.
        """
        if not other.uniformly_spaced:
            raise NotImplementedError("only uniformly spaced grids are supported.")
        if not other.uniformly_weighted:
            raise NotImplementedError("only uniformly weighted grids are supported.")
        if other.domain != "time" or other.reality:
            raise ValueError(f"dual grid for {cls.__name__} must be of type ImaginaryTimeGrid")
        return cls.from_uniform(len(other) // 2, other.beta)  # Use τ:ω ratio of 2:1


GridIF = ImaginaryFrequencyGrid
