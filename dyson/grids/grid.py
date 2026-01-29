"""Base class for grids."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
from typing_extensions import Self

from dyson import numpy as np
from dyson import util
from dyson.representations.enums import Component, Reduction, RepresentationEnum, Ordering

if TYPE_CHECKING:
    from typing import Any, Iterable

    from dyson.representations.dynamic import Dynamic
    from dyson.representations.lehmann import Lehmann
    from dyson.typing import Array


class BaseGrid(ABC):
    """Base class for grids."""

    _options: set[str] = set()

    _points: Array
    _weights: Array | None = None

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

    def set_options(self, **kwargs: Any) -> None:
        """Set options for the solver.

        Args:
            kwargs: Keyword arguments to set as options.
        """
        for key, val in kwargs.items():
            if key not in self._options:
                raise ValueError(f"Unknown option for {self.__class__.__name__}: {key}")
            if isinstance(getattr(self, key), RepresentationEnum):
                # Casts string to the appropriate enum type if the default value is an enum
                val = getattr(self, key).__class__(val)
            setattr(self, key, val)

    @abstractmethod
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
            ordering: Time ordering of the resolvent or propagator.

        Returns:
            Kernel of a Lehmann representation on the grid.

        Note:
            The kernel is a hook to generalise the resolvent or propagator, depending on whether
            the grid is in the frequency or time domain.
        """
        pass

    def evaluate_lehmann(
        self,
        lehmann: Lehmann,
        reduction: Reduction = Reduction.NONE,
        component: Component = Component.FULL,
        ordering: Ordering = Ordering.ORDERED,
    ) -> Dynamic[Any]:
        r"""Evaluate a Lehmann representation on the grid.

        The imaginary frequency representation is defined as

        .. math::
            \sum_{k} \frac{v_{pk} u_{qk}^*}{i \omega - \epsilon_k},

        and the real frequency representation is defined as

        .. math::
            \sum_{k} \frac{v_{pk} u_{qk}^*}{\omega - \epsilon_k \pm i \eta},

        where :math:`\omega` is the frequency grid, :math:`\epsilon_k` are the poles, and the sign
        of the broadening factor is determined by the time ordering.

        Args:
            lehmann: Lehmann representation to evaluate.
            reduction: The reduction of the dynamic representation.
            component: The component of the dynamic representation.
            ordering: The time ordering of the Lehmann representation.

        Returns:
            Lehmann representation, realised on the grid.
        """
        from dyson.representations.dynamic import Dynamic  # noqa: PLC0415

        left, right = lehmann.unpack_couplings()
        kernel = self._lehmann_kernel(lehmann.energies, lehmann.chempot, ordering=ordering)
        reduction = Reduction(reduction)
        component = Component(component)
        ordering = Ordering(ordering)

        # Get the input and output indices based on the reduction type
        inp = "qk"
        out = "wpq"
        if reduction == reduction.NONE:
            pass
        elif reduction == reduction.DIAG:
            inp = "pk"
            out = "wp"
        elif reduction == reduction.TRACE:
            inp = "pk"
            out = "w"
        else:
            reduction.raise_invalid_representation()

        # Perform the downfolding operation
        array = util.einsum(f"pk,{inp},wk->{out}", right, left.conj(), kernel)

        # Get the required component
        # TODO: Save time by not evaluating the full array when not needed
        if component == Component.REAL:
            array = array.real
        elif component == Component.IMAG:
            array = array.imag

        return Dynamic(
            self,
            array,
            reduction=reduction,
            component=component,
            ordering=ordering,
            hermitian=lehmann.hermitian,
        )

    @classmethod
    @abstractmethod
    def from_dual(cls, other: BaseGrid) -> Self:
        """Create a grid from another grid in the dual domain.

        Args:
            other: Other grid to create from.

        Returns:
            Grid.
        """
        pass

    @property
    def points(self) -> Array:
        """Get the points of the grid.

        Returns:
            Points of the grid.
        """
        return self._points

    @property
    def weights(self) -> Array:
        """Get the weights of the grid.

        Returns:
            Weights of the grid.
        """
        if self._weights is None:
            return np.ones_like(self.points) / len(self)
        return self._weights

    def __getitem__(self, key: int | slice | list[int] | Array) -> BaseGrid:
        """Get a subset of the grid.

        Args:
            key: Index or slice to get.

        Returns:
            Subset of the grid.
        """
        points = self.points[key]
        weights = self.weights[key] if self._weights is not None else None
        kwargs = {opt: getattr(self, opt) for opt in self._options}
        return self.__class__(points, weights=weights, **kwargs)

    def __len__(self) -> int:
        """Get the size of the grid.

        Returns:
            Size of the grid.
        """
        return self.points.shape[0]

    @property
    def uniformly_spaced(self) -> bool:
        """Check if the grid is uniformly spaced.

        Returns:
            True if the grid is uniformly spaced, False otherwise.
        """
        if len(self) < 2:
            raise ValueError("Grid is too small to compute separation.")
        return np.allclose(np.diff(self.points), self.points[1] - self.points[0])

    @property
    def uniformly_weighted(self) -> bool:
        """Check if the grid is uniformly weighted.

        Returns:
            True if the grid is uniformly weighted, False otherwise.
        """
        return np.allclose(self.weights, self.weights[0])

    @property
    def separation(self) -> float:
        """Get the separation of the grid.

        Returns:
            Separation of the grid.
        """
        if not self.uniformly_spaced:
            raise ValueError("Grid is not uniformly spaced.")
        return np.abs(self.points[1] - self.points[0])

    @property
    @abstractmethod
    def domain(self) -> str:
        """Get the domain of the grid."""
        pass

    @property
    @abstractmethod
    def reality(self) -> bool:
        """Get the reality of the grid."""
        pass
