"""Real frequency Dyson equation solver."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing_extensions import Self

from dyson import console, printing, util
from dyson import numpy as np
from dyson.representations.dynamic import Dynamic
from dyson.representations.enums import Component, Ordering, Reduction
from dyson.solvers.solver import DynamicSolver
from dyson.solvers.recipes import greens_function_from_hamiltonian
from dyson.grids.util import are_equal

if TYPE_CHECKING:
    from typing import Any

    from dyson.grids.frequency import BaseFrequencyGrid
    from dyson.expressions.expression import BaseExpression
    from dyson.representations.lehmann import Lehmann
    from dyson.typing import Array


class Direct(DynamicSolver):
    """Direct frequency-space Dyson equation solver."""

    def __init__(  # noqa: D417
        self,
        greens_function_init: Dynamic[BaseFrequencyGrid],
        self_energy: Dynamic[BaseFrequencyGrid],
        overlap: Array | None = None,
        **kwargs: Any,
    ):
        """Initialise the solver.

        Args:
            greens_function_init: The initial Green's function (i.e. :math:`G_0`).
            self_energy: The self-energy to solve.
        """
        self._greens_function_init = greens_function_init
        self._self_energy = self_energy
        self._overlap = overlap if overlap is not None else np.eye(greens_function_init.nphys)
        self.set_options(**kwargs)

    def __post_init__(self) -> None:
        """Hook called after :meth:`__init__`."""
        # Check the input
        if self.greens_function_init.nphys != self.self_energy.nphys:
            raise ValueError("input functions must have the same number of physical indices")
        if not are_equal(self.greens_function_init.grid, self.self_energy.grid):
            raise ValueError("input functions must be defined on the same grid")
        if self.greens_function_init.grid.domain != "frequency":
            raise ValueError("input functions must be defined on a frequency grid")
        if self.greens_function_init.reduction != self.self_energy.reduction:
            raise ValueError("input functions must have the same reduction")
        if self.greens_function_init.component != self.self_energy.component:
            raise ValueError("input functions must have the same component")
        if self.greens_function_init.ordering != self.self_energy.ordering:
            raise ValueError("input functions must have the same ordering")

        # Warn if ordering is time-ordered, which may be numerically unstable
        if self.greens_function_init.ordering == Ordering.ORDERED:
            console.print(
                "[bad]Input functions are time-ordered, which may lead to numerical instability "
                "near poles. Consider using retarded or advanced ordering if possible.[/bad]"
            )

        # Print the input informationn
        cond = printing.format_float(
            np.linalg.cond(self.overlap), threshold=1e10, scientific=True, precision=4
        )
        console.print(f"Number of physical states: [input]{self.nphys}[/input]")
        console.print(f"Overlap condition number: {cond}")

    @classmethod
    def from_self_energy(
        cls,
        static: Array,
        self_energy: Lehmann,
        overlap: Array | None = None,
        **kwargs: Any,
    ) -> Self:
        """Create a solver from a self-energy.

        Args:
            static: Static part of the self-energy.
            self_energy: Self-energy.
            overlap: Overlap matrix for the physical space.
            kwargs: Additional keyword arguments for the solver.

        Returns:
            Solver instance.
        """
        if "grid" not in kwargs:
            raise ValueError("Missing required argument grid.")
        grid: BaseFrequencyGrid = kwargs.pop("grid")
        representation = dict(
            ordering=kwargs.pop("ordering", Ordering.ORDERED),
            reduction=kwargs.pop("reduction", Reduction.NONE),
            component=kwargs.pop("component", Component.FULL),
        )
        greens_function_init = grid.evaluate_lehmann(
            greens_function_from_hamiltonian(static, overlap=overlap), **representation,
        )
        self_energy_grid = grid.evaluate_lehmann(self_energy, **representation)
        return cls(
            greens_function_init=greens_function_init,
            self_energy=self_energy_grid,
            overlap=overlap,
            **kwargs,
        )

    @classmethod
    def from_expression(cls, expression: BaseExpression, **kwargs: Any) -> Self:
        """Create a solver from an expression.

        Args:
            expression: Expression to be solved.
            kwargs: Additional keyword arguments for the solver.

        Returns:
            Solver instance.
        """
        raise NotImplementedError(
            f"Cannot instantiate {cls.__name__} from an expression. "
            f"Please use {cls.__name__}.from_self_energy instead."
        )

    def kernel(self) -> Dynamic[BaseFrequencyGrid]:
        """Run the solver.

        Returns:
            The Green's function on the frequency grid.
        """
        return (self.greens_function_init.inverse() - self.self_energy).inverse()

    @property
    def greens_function_init(self) -> Dynamic[BaseFrequencyGrid]:
        """Get the initial Green's function."""
        return self._greens_function_init

    @property
    def self_energy(self) -> Dynamic[BaseFrequencyGrid]:
        """Get the self-energy."""
        return self._self_energy

    @property
    def overlap(self) -> Array:
        """Get the overlap matrix."""
        return self._overlap

    @property
    def grid(self) -> BaseFrequencyGrid:
        """Get the frequency grid."""
        return self.greens_function_init.grid

    @property
    def nphys(self) -> int:
        """Get the number of physical degrees of freedom."""
        return self.greens_function_init.nphys
