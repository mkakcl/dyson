"""Fourier transformation."""

from __future__ import annotations

from typing import TYPE_CHECKING
import warnings

from dyson import util
from dyson import numpy as np
from dyson.grids.util import are_dual
from dyson.representations.enums import Component, Reduction

if TYPE_CHECKING:
    from dyson.representations.dynamic import Dynamic
    from dyson.grids.grid import BaseGrid
    from dyson.grids.frequency import GridIF
    from dyson.grids.time import GridIT
    from dyson.typing import Array


def fourier_transform_imag(greens_function: Dynamic[BaseGrid], grid: BaseGrid) -> Dynamic[BaseGrid]:
    """Fourier transform between imaginary frequency and imaginary time grids.

    Args:
        greens_function: Dynamic quantity in imaginary frequency or time domain.
        grid: Target grid for the Fourier transform.

    Returns:
        Dynamic quantity in the target domain.
    """
    grid_in, grid_out = greens_function.grid, grid
    if not are_dual(grid_in, grid_out):
        raise ValueError("the two grids must be dual to each other.")
    if grid_in.reality or grid_out.reality:
        raise ValueError("only imaginary frequency and imaginary time grids is supported.")
    if greens_function.component != Component.FULL:
        raise ValueError("only full component is supported.")
    if greens_function.reduction == Reduction.TRACE:
        raise ValueError("traced reduction is not supported.")
    forward = grid_in.domain == "time"

    # Setup based on direction
    beta = grid_in.beta
    if forward:
        sign = 1
        norm = beta
        fft = np.fft.ifft
    else:
        sign = -1
        norm = 2 / beta
        fft = np.fft.fft

    # Get the shifts
    shifts = (
        np.exp(1.0j * sign * np.min(grid_out.points) * grid_in.points),
        np.exp(1.0j * sign * np.min(grid_in.points) * (grid_out.points - np.min(grid_out.points))),
    )

    # Get the input array
    array = greens_function.array.copy()

    # Perform the Fourier transform
    array = util.einsum("w...,w->w...", array, shifts[0])
    array = fft(array, max(len(grid_in), len(grid_out)), axis=0)[: len(grid_out)]
    array = util.einsum("w...,w->w...", array, shifts[1])

    # Normalise
    array *= norm

    # Hermitise for imaginary frequency to imaginary time transform
    if not forward:
        if greens_function.reduction == Reduction.NONE:
            array = 0.5 * (array + array.swapaxes(1, 2).conj())
        else:
            array = 0.5 * (array + array.conj())

    return greens_function.__class__(
        grid_out,
        array,
        reduction=greens_function.reduction,
        ordering=greens_function.ordering,
        hermitian=greens_function.hermitian,
    )
