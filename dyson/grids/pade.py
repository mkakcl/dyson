"""Pade method for analytic continuation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from dyson import util
from dyson import numpy as np
from dyson.representations.enums import Ordering, Component, Reduction

if TYPE_CHECKING:
    from dyson.representations.dynamic import Dynamic
    from dyson.grids.frequency import GridIF, GridRF, BaseFrequencyGrid
    from dyson.typing import Array


def _pade_coefficients(greens_function: Dynamic[BaseFrequencyGrid]) -> Array:
    """Get the coefficient for the Pade approximation for a frequency domain function.

    Args:
        greens_function: Dynamic quantity in frequency domain.

    Returns:
        Coefficients for the Pade approximation.
    """
    grid = greens_function.grid
    if not grid.uniformly_spaced:
        raise ValueError("only uniform grids are supported.")
    if not grid.uniformly_weighted:
        raise ValueError("only uniform weights are supported.")

    # Initialise the coefficients
    resolvent = grid.resolvent(np.array(0.0), 0.0, invert=False, ordering=greens_function.ordering)
    coefficients = greens_function.array.copy()

    # Recursively compute the coefficients
    for i in range(len(grid) - 1):
        with np.errstate(divide="ignore", invalid="ignore"):
            factor = coefficients[i] / coefficients[i + 1 :] - 1.0
        factor[~np.isfinite(factor)] = 0.0
        difference = resolvent[i + 1 :] - resolvent[i]
        if (np.iscomplexobj(factor) or np.iscomplexobj(difference)) and not np.iscomplexobj(coefficients):
            coefficients = coefficients.astype(np.complex128)
        coefficients[i + 1 :] = util.einsum("w...,w->w...", factor, 1.0 / difference)

    return coefficients


def evaluate_pade(
    coefficients: Array,
    grid_old: BaseFrequencyGrid,
    grid_new: BaseFrequencyGrid,
    ordering: Ordering = Ordering.RETARDED,
) -> Array:
    """Evaluate the Pade approximation on a new frequency grid.

    Args:
        coefficients: Coefficients for the Pade approximation.
        grid_old: Original frequency grid.
        grid_new: New frequency grid.
        ordering: Ordering of the Green's function.

    Returns:
        Dynamic quantity on the new frequency grid.
    """
    if not grid_old.uniformly_spaced or not grid_new.uniformly_spaced:
        raise ValueError("only uniform grids are supported.")
    if not grid_old.uniformly_weighted or not grid_new.uniformly_weighted:
        raise ValueError("only uniform weights are supported.")

    # Initialise the array
    resolvent_old = grid_old.resolvent(np.array(0.0), 0.0, invert=False, ordering=ordering)
    resolvent_new = grid_new.resolvent(np.array(0.0), 0.0, invert=False, ordering=ordering)
    array = coefficients[[-1]].copy()

    # Recursively evaluate the Pade approximation
    for i in range(len(grid_old) - 1, -1, -1):
        term = 1.0 + util.einsum("w,w...->w...", resolvent_new - resolvent_old[i], array)
        with np.errstate(divide="ignore", invalid="ignore"):
            array = coefficients[i] / term
        array[~np.isfinite(array)] = 0.0

    return array


def analytic_continuation_freq_pade(
    greens_function: Dynamic[BaseFrequencyGrid],
    grid: BaseFrequencyGrid,
) -> Dynamic[BaseFrequencyGrid]:
    """Perform analytic continuation in the frequency domain using the Pade approximation.

    Args:
        greens_function_if: Green's function in a frequency domain.
        grid_rf: Real frequency grid to continue to.

    Returns:
        Green's function in the conjugate frequency domain.
    """
    if greens_function.ordering == Ordering.ORDERED:
        raise ValueError("time-ordered ordering is not supported.")
    if greens_function.component != Component.FULL:
        raise ValueError("only full component is supported.")
    if greens_function.reduction == Reduction.TRACE:
        raise ValueError("traced reduction is not supported.")
    coefficients = _pade_coefficients(greens_function)
    array = evaluate_pade(
        coefficients,
        greens_function.grid,
        grid,
        ordering=greens_function.ordering,
    )
    return greens_function.__class__(
        grid,
        array,
        reduction=greens_function.reduction,
        hermitian=greens_function.hermitian,
        ordering=greens_function.ordering,
    )
