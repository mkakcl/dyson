"""Common functionality for moment block Lanczos solvers."""

from __future__ import annotations

import dataclasses
import functools
import math
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from dyson import console, printing, util
from dyson import numpy as np
from dyson.representations.enums import Reduction
from dyson.solvers.solver import StaticSolver

if TYPE_CHECKING:
    from dyson.representations.lehmann import Lehmann
    from dyson.representations.spectral import Spectral
    from dyson.typing import Array

# TODO: reimplement caching


def _relative(error: float, scale: float) -> float:
    """Divide an error by the scale of its reference, handling a zero reference.

    Args:
        error: The absolute error.
        scale: The norm of the reference quantity.

    Returns:
        The relative error. A zero reference gives zero if the error is also zero, and infinity
        otherwise, rather than a ``nan`` that would silently propagate through a comparison.
    """
    if scale > 0.0:
        return error / scale
    return 0.0 if error == 0.0 else float(np.inf)


def _error_norms(predicted: Array, reference: Array) -> tuple[float, float, float, float]:
    """Compute the error norms between a single predicted and reference moment.

    Both norms are entrywise, so they are well defined for the matrix-valued moments of
    :cls:`MBLSE` and :cls:`MBLGF` and the scalar-valued moments of :cls:`MLSE` alike.

    Args:
        predicted: The reconstructed moment.
        reference: The reference moment.

    Returns:
        The absolute and relative Frobenius norms, then the absolute and relative maximum
        norms, of the difference.
    """
    difference = np.ravel(np.asarray(predicted) - np.asarray(reference))
    scale = np.ravel(np.asarray(reference))

    absolute_frobenius = float(np.linalg.norm(difference))
    absolute_max = float(np.max(np.abs(difference))) if difference.size else 0.0
    scale_frobenius = float(np.linalg.norm(scale))
    scale_max = float(np.max(np.abs(scale))) if scale.size else 0.0

    return (
        absolute_frobenius,
        _relative(absolute_frobenius, scale_frobenius),
        absolute_max,
        _relative(absolute_max, scale_max),
    )


def shift_moments(moments: Array, shift: float) -> Array:
    r"""Shift the poles of a set of moments by a constant.

    Applies the binomial transform that maps moments taken about the origin onto moments taken
    about ``shift``,

    .. math::
        \tilde{T}^{n} = \sum_{k=0}^{n} \binom{n}{k} (-\mu)^{n-k} T^{k},

    which is exact and needs only the moments themselves, not the underlying poles.

    Args:
        moments: The moments, indexed by order along the leading axis.
        shift: The amount to shift the poles by, :math:`\mu`.

    Returns:
        The shifted moments.
    """
    moments = np.asarray(moments)
    shifted = np.zeros_like(moments)
    for n in range(moments.shape[0]):
        for k in range(n + 1):
            shifted[n] += math.comb(n, k) * (-shift) ** (n - k) * moments[k]
    return shifted


def _compare_moments(predicted: Array, reference: Array, orders: tuple[int, ...]) -> MomentErrors:
    """Compare two equal-length sets of moments order by order.

    Args:
        predicted: The reconstructed moments.
        reference: The reference moments.
        orders: The moment orders being compared.

    Returns:
        The per-order errors.
    """
    norms = [_error_norms(p, r) for p, r in zip(predicted, reference)]
    return MomentErrors(
        orders=orders,
        absolute_frobenius=tuple(n[0] for n in norms),
        relative_frobenius=tuple(n[1] for n in norms),
        absolute_max=tuple(n[2] for n in norms),
        relative_max=tuple(n[3] for n in norms),
        scaled=tuple(util.scaled_error(p, r) for p, r in zip(predicted, reference)),
    )


@dataclasses.dataclass(frozen=True)
class MomentErrors:
    """Per-order errors between reconstructed and reference moments.

    Args:
        orders: The moment orders the errors correspond to.
        absolute_frobenius: Absolute Frobenius-norm error at each order.
        relative_frobenius: Frobenius-norm error at each order, relative to the reference.
        absolute_max: Absolute maximum-norm error at each order.
        relative_max: Maximum-norm error at each order, relative to the reference.
        scaled: The :func:`~dyson.util.linalg.scaled_error` at each order, retained so the
            aggregate reported during the recurrence keeps its historical meaning.
        chempot: The chemical potential of the reconstructed representation.
        shifted: If :attr:`chempot` is non-zero, the same errors recomputed with both sets of
            moments shifted onto poles measured from :attr:`chempot`. ``None`` when no shift
            applies, in which case the shifted errors would equal the unshifted ones.
    """

    orders: tuple[int, ...]
    absolute_frobenius: tuple[float, ...]
    relative_frobenius: tuple[float, ...]
    absolute_max: tuple[float, ...]
    relative_max: tuple[float, ...]
    scaled: tuple[float, ...]
    chempot: float = 0.0
    shifted: MomentErrors | None = None

    @property
    def total(self) -> float:
        """Get the summed scaled error over all orders."""
        return float(sum(self.scaled))

    @property
    def max_relative_frobenius(self) -> float:
        """Get the largest relative Frobenius-norm error over all orders."""
        return max(self.relative_frobenius, default=0.0)

    @property
    def max_relative_max(self) -> float:
        """Get the largest relative maximum-norm error over all orders."""
        return max(self.relative_max, default=0.0)

    def print(self, title: str = "Moment errors") -> None:
        """Print the per-order errors as a table.

        Args:
            title: The title for the table.
        """
        table = printing.Table(title=title, box=printing.box.SIMPLE)
        table.add_column("Order", style="dim", justify="left")
        for column in ("Abs. Frobenius", "Rel. Frobenius", "Abs. max", "Rel. max"):
            table.add_column(column, justify="right")
        for i, order in enumerate(self.orders):
            table.add_row(
                str(order),
                printing.format_float(self.absolute_frobenius[i], scientific=True, precision=4),
                printing.format_float(
                    self.relative_frobenius[i], threshold=1e-10, scientific=True, precision=4
                ),
                printing.format_float(self.absolute_max[i], scientific=True, precision=4),
                printing.format_float(
                    self.relative_max[i], threshold=1e-10, scientific=True, precision=4
                ),
            )
        console.print(table)

        if self.shifted is not None:
            chempot = printing.format_float(self.chempot, precision=6)
            self.shifted.print(title=f"{title} (poles shifted by chempot {chempot})")


class BaseRecursionCoefficients(ABC):
    """Base class for recursion coefficients for the moment block Lanczos algorithms.

    Args:
        nphys: Number of physical degrees of freedom.
    """

    NDIM: int = 2

    def __init__(
        self,
        nphys: int,
        hermitian: bool = True,
        force_orthogonality: bool = True,
    ):
        """Initialise the recursion coefficients."""
        self._nphys = nphys
        self._zero = np.zeros((nphys,) * self.NDIM)
        self._data: dict[tuple[int, ...], Array] = {}
        self.hermitian = hermitian
        self.force_orthogonality = force_orthogonality

    @property
    def nphys(self) -> int:
        """Get the number of physical degrees of freedom."""
        return self._nphys

    @property
    def dtype(self) -> str:
        """Get the data type of the recursion coefficients."""
        if any([np.iscomplexobj(v) for v in self._data.values()]):
            return "complex128"
        return "float64"

    @abstractmethod
    def __getitem__(self, key: tuple[int, ...]) -> Array:
        """Get the recursion coefficients for the given key.

        Args:
            key: The key for the recursion coefficients.

        Returns:
            The recursion coefficients.
        """
        pass

    @abstractmethod
    def __setitem__(self, key: tuple[int, ...], value: Array) -> None:
        """Set the recursion coefficients for the given key.

        Args:
            key: The key for the recursion coefficients.
            value: The recursion coefficients.
        """
        pass


class BaseMBL(StaticSolver):
    """Base class for moment block Lanczos solvers."""

    Coefficients: type[BaseRecursionCoefficients]

    _moments: Array

    max_cycle: int
    hermitian: bool = True
    force_orthogonality: bool = True
    calculate_errors: bool = True
    _options: set[str] = {"max_cycle", "hermitian", "force_orthogonality", "calculate_errors"}

    #: The reduction the solver's moments are expressed in, so that reconstructed and reference
    #: moments are always compared in the same representation.
    moment_reduction: Reduction = Reduction.NONE

    #: The highest iteration the recurrence actually completed. Equal to :attr:`max_cycle`
    #: unless the input could not support the requested order, in which case the recurrence
    #: stopped early and this records where.
    max_cycle_achieved: int | None = None

    def __post_kernel__(self) -> None:
        """Hook called after :meth:`kernel`."""
        assert self.result is not None
        emin = printing.format_float(self.result.eigvals.min())
        emax = printing.format_float(self.result.eigvals.max())
        console.print(
            f"Found [output]{self.result.neig}[/output] roots between [output]{emin}[/output] and "
            f"[output]{emax}[/output]."
        )
        if self.max_cycle_achieved is not None and self.max_cycle_achieved != self.max_cycle:
            console.print(
                f"Moment orders: requested [input]{self.nmom_conserved(self.max_cycle)}[/input], "
                f"realized [output]{self.nmom_conserved(self.max_cycle_achieved)}[/output]."
            )
        if self.calculate_errors:
            errors = self.moment_errors(iteration=self.max_cycle_achieved)
            total = printing.format_float(
                errors.total, threshold=1e-10, scientific=True, precision=4
            )
            console.print(
                f"Error in the moments: {total} (over {len(errors.orders)} conserved orders)"
            )
            errors.print()

    def validate_moments(self) -> None:
        """Check that the supplied moments can define a measure before the recurrence starts.

        Raises:
            ValueError: If the moments are not finite, are not Hermitian when the solver has been
                told they are, or if the zeroth moment is not positive semi-definite. The zeroth
                moment is a sum of outer products of couplings, so a negative direction in it is
                negative spectral weight: the input does not describe a measure and no amount of
                recurrence will make it one.
        """
        moments = np.asarray(self.moments)

        if not np.all(np.isfinite(moments)):
            bad = int(np.count_nonzero(~np.isfinite(moments)))
            raise ValueError(
                f"Moments contain {bad} non-finite element(s). The recurrence cannot start."
            )

        if self.hermitian and moments.ndim == 3:
            asymmetry = max(
                float(np.max(np.abs(moment - moment.T.conj()))) if moment.size else 0.0
                for moment in moments
            )
            scale = float(np.max(np.abs(moments))) if moments.size else 0.0
            if asymmetry > 1e-10 + 1e-8 * scale:
                raise ValueError(
                    f"Moments are not Hermitian to within tolerance: the largest deviation from "
                    f"the Hermitian conjugate is {asymmetry:.6e} against a largest element of "
                    f"{scale:.6e}. Pass hermitian=False if this is intended."
                )

        # The zeroth moment carries the total spectral weight and must be positive semi-definite.
        zeroth = np.atleast_2d(moments[0])
        if self.hermitian and not np.iscomplexobj(zeroth):
            eigvals = np.real(np.linalg.eigvalsh(zeroth))
            scale = float(np.max(np.abs(eigvals))) if eigvals.size else 0.0
            worst = float(np.min(eigvals)) if eigvals.size else 0.0
            if worst < -(1e-10 + 1e-8 * scale):
                raise ValueError(
                    f"The zeroth moment is not positive semi-definite: its smallest eigenvalue is "
                    f"{worst:.6e} against a largest of {scale:.6e}. This is negative spectral "
                    "weight in the input, not an artefact of the recurrence."
                )

    def report_moment_usage(self) -> None:
        """Report how many of the supplied moments the requested order actually consumes.

        The recurrence consumes ``2 * max_cycle + 2`` moments, so an odd number of supplied
        moments always leaves one unused. Saying so is the alternative to requiring a particular
        parity: the caller learns exactly which moments were used rather than silently losing one.
        """
        supplied = int(np.shape(self.moments)[0])
        used = self.nmom_conserved(self.max_cycle)
        if used == supplied:
            return
        console.print(
            f"Using [output]{used}[/output] of [input]{supplied}[/input] supplied moments "
            f"(orders 0-{used - 1}); the recurrence consumes 2 * max_cycle + 2 and the "
            f"remaining {supplied - used} cannot be constrained at max_cycle="
            f"[input]{self.max_cycle}[/input]."
        )

    @abstractmethod
    def solve(self, iteration: int | None = None) -> Spectral:
        """Solve the eigenvalue problem at a given iteration.

        Args:
            iteration: The iteration to get the results for.

        Returns:
            The :cls:`Spectral` object.
        """
        pass

    def kernel(self) -> Spectral:
        """Run the solver.

        Returns:
            The eigenvalues and eigenvectors of the self-energy supermatrix.
        """
        # Refuse input that cannot describe a measure before doing any work on it
        self.validate_moments()
        self.report_moment_usage()

        # Get the table
        table = printing.ConvergencePrinter(
            (), ("Error in moments", "Error in sqrt", "Error in inv. sqrt"), (1e-10, 1e-10, 1e-10)
        )
        progress = printing.IterationsPrinter(self.max_cycle)
        progress.start()

        # Run the solver
        self.max_cycle_achieved = self.max_cycle
        for iteration in range(self.max_cycle + 1):  # TODO: check
            try:
                error_sqrt, error_inv_sqrt, error_moments = self.recurrence_iteration(iteration)
            except util.NotPositiveSemiDefiniteError as error:
                # The block this iteration would add has no real square root, so the requested
                # order is not supportable by these moments. Everything through the previous
                # iteration is complete and untouched, so step down to it rather than either
                # failing outright or silently discarding the offending direction.
                if iteration == 0:
                    raise
                self.max_cycle_achieved = iteration - 1
                progress.stop()
                table.print()
                console.print(
                    f"[bad]Requested order {self.max_cycle} is not realizable[/bad]: iteration "
                    f"{iteration} produced an off-diagonal square that is not positive "
                    f"semi-definite. Stepping down to order "
                    f"[output]{self.max_cycle_achieved}[/output], conserving "
                    f"[output]{self.nmom_conserved(self.max_cycle_achieved)}[/output] of the "
                    f"{self.nmom_conserved(self.max_cycle)} moments requested."
                )
                console.print(f"Cause: {error}")
                break
            if not self.calculate_errors:
                error_sqrt = error_inv_sqrt = error_moments = np.nan
            table.add_row(iteration, (), (error_moments, error_sqrt, error_inv_sqrt))
            progress.update(iteration)
        else:
            progress.stop()
            table.print()

        # Diagonalise the compressed self-energy
        self.result = self.solve(iteration=self.max_cycle_achieved)

        return self.result

    @functools.cached_property
    def orthogonalisation_metric(self) -> Array:
        """Get the orthogonalisation metric."""
        return util.matrix_power(self.moments[0], -0.5, hermitian=self.hermitian, strict=True)[0]

    @functools.cached_property
    def orthogonalisation_metric_inv(self) -> Array:
        """Get the inverse of the orthogonalisation metric."""
        return util.matrix_power(self.moments[0], 0.5, hermitian=self.hermitian, strict=True)[0]

    @functools.lru_cache(maxsize=64)
    def orthogonalised_moment(self, order: int) -> Array:
        """Compute an orthogonalised moment.

        Args:
            order: The order of the moment.

        Returns:
            The orthogonalised moment.
        """
        return self.orthogonalisation_metric @ self.moments[order] @ self.orthogonalisation_metric

    @staticmethod
    def nmom_conserved(iteration: int) -> int:
        """Get the number of moments conserved by the recurrence at a given iteration.

        A block Lanczos recurrence run to ``iteration`` conserves ``2 * iteration + 2`` moments.
        This is the single definition of that count: it fixes both how many moments are
        reconstructed and how many are used as a reference, so the two cannot disagree.

        Args:
            iteration: The iteration number.

        Returns:
            The number of moments conserved.
        """
        return 2 * iteration + 2

    @abstractmethod
    def reconstruct_representation(self, iteration: int) -> Lehmann:
        """Reconstruct the Lehmann representation implied by the recurrence.

        Args:
            iteration: The iteration number.

        Returns:
            The reconstructed representation.
        """
        pass

    def reconstruct_moments(self, iteration: int) -> Array:
        """Reconstruct the moments conserved at a given iteration.

        Args:
            iteration: The iteration number.

        Returns:
            The reconstructed moments, one for each conserved order.
        """
        representation = self.reconstruct_representation(iteration)
        orders = range(self.nmom_conserved(iteration))
        return representation.moments(orders, reduction=self.moment_reduction)

    def reference_moments(self, iteration: int) -> Array:
        """Get the input moments that the reconstruction at a given iteration must reproduce.

        Args:
            iteration: The iteration number.

        Returns:
            The reference moments, one for each conserved order.
        """
        return self.moments[: self.nmom_conserved(iteration)]

    def moment_errors(self, iteration: int | None = None) -> MomentErrors:
        """Get the per-order moment errors at a given iteration.

        Args:
            iteration: The iteration to check. Defaults to :attr:`max_cycle`.

        Returns:
            The per-order errors, including the errors after any chemical-potential pole shift.

        Raises:
            ValueError: If the number of reconstructed moments differs from the number of
                reference moments, which would otherwise let a silently truncated comparison
                report a small error over a subset of the moments.
        """
        if iteration is None:
            iteration = (
                self.max_cycle if self.max_cycle_achieved is None else self.max_cycle_achieved
            )

        representation = self.reconstruct_representation(iteration)
        orders = range(self.nmom_conserved(iteration))
        predicted = representation.moments(orders, reduction=self.moment_reduction)
        reference = self.reference_moments(iteration)

        if len(predicted) != len(reference):
            raise ValueError(
                f"Reconstructed {len(predicted)} moments at iteration {iteration} but have "
                f"{len(reference)} reference moments. The recurrence conserves "
                f"{self.nmom_conserved(iteration)} moments, so these must agree; comparing "
                "them would silently ignore the surplus."
            )

        errors = _compare_moments(predicted, reference, tuple(orders))

        # Report the errors again with the poles measured from the chemical potential, so that a
        # shift can be seen to be a change of convention rather than a source of error.
        chempot = float(getattr(representation, "chempot", 0.0) or 0.0)
        if chempot:
            shifted = _compare_moments(
                shift_moments(predicted, chempot),
                shift_moments(reference, chempot),
                tuple(orders),
            )
            errors = dataclasses.replace(errors, chempot=chempot, shifted=shifted)

        return errors

    def moment_error(self, iteration: int | None = None) -> float:
        """Get the aggregate moment error at a given iteration.

        Args:
            iteration: The iteration to check. Defaults to :attr:`max_cycle`.

        Returns:
            The summed scaled error over every conserved order. Use :meth:`moment_errors` for
            the per-order breakdown.
        """
        return self.moment_errors(iteration=iteration).total

    @abstractmethod
    def initialise_recurrence(self) -> tuple[float | None, float | None, float | None]:
        """Initialise the recurrence (zeroth iteration).

        Returns:
            If :attr:`calculate_errors`, the error metrics in the square root of the off-diagonal
            block, the inverse square root of the off-diagonal block, and the error in the
            recovered moments. If not, all three are ``None``.
        """
        pass

    @abstractmethod
    def _recurrence_iteration_hermitian(
        self, iteration: int
    ) -> tuple[float | None, float | None, float | None]:
        """Perform an iteration of the recurrence for a Hermitian self-energy."""
        pass

    @abstractmethod
    def _recurrence_iteration_non_hermitian(
        self, iteration: int
    ) -> tuple[float | None, float | None, float | None]:
        """Perform an iteration of the recurrence for a non-Hermitian self-energy."""
        pass

    def recurrence_iteration(
        self, iteration: int
    ) -> tuple[float | None, float | None, float | None]:
        """Perform an iteration of the recurrence.

        Args:
            iteration: The iteration to perform.

        Returns:
            If :attr:`calculate_errors`, the error metrics in the square root of the off-diagonal
            block, the inverse square root of the off-diagonal block, and the error in the
            recovered moments. If not, all three are ``None``.
        """
        if iteration == 0:
            return self.initialise_recurrence()
        if iteration > self.max_cycle:
            raise ValueError(f"Iteration {iteration} exceeds max_cycle {self.max_cycle}.")
        if self.hermitian:
            return self._recurrence_iteration_hermitian(iteration)
        return self._recurrence_iteration_non_hermitian(iteration)

    @property
    def moments(self) -> Array:
        """Get the moments of the self-energy."""
        return self._moments

    @property
    def nphys(self) -> int:
        """Get the number of physical degrees of freedom."""
        return self.moments.shape[-1]
