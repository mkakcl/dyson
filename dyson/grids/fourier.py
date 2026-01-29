"""Fourier transformation."""

from __future__ import annotations

from typing import TYPE_CHECKING
import warnings

from dyson import util
from dyson import numpy as np
from dyson.grids.util import are_dual
from dyson.representations.enums import Component

if TYPE_CHECKING:
    from dyson.representations.dynamic import Dynamic
    from dyson.grids.grid import BaseGrid
    from dyson.grids.frequency import GridIF
    from dyson.grids.time import GridIT
    from dyson.typing import Array


def fourier_transform_imag(
    greens_function: Dynamic[BaseGrid],
    grid: BaseGrid,
    tail_moments: tuple[Array, ...] | None = None,
) -> Dynamic[BaseGrid]:
    """Fourier transform between imaginary frequency and imaginary time grids.

    Args:
        greens_function: Dynamic quantity in imaginary frequency or time domain.
        grid: Target grid for the Fourier transform.
        tail_moments: Moments of the high-frequency tail.

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
    forward = grid_in.domain == "time"

    # Setup based on direction
    beta = grid_in.beta
    if forward:
        freqs, times = grid_out.points, grid_in.points
        sign = 1
        norm = beta
        fft = np.fft.ifft
    else:
        freqs, times = grid_in.points, grid_out.points
        sign = -1
        norm = 1 / beta
        fft = np.fft.fft

    # Get the shifts
    shift_if = np.exp(1.0j * sign * np.pi * times / beta)
    shift_it = np.exp(1.0j * sign * (freqs - np.pi / beta) * times[0])
    shifts = (shift_if, shift_it) if forward else (shift_it, shift_if)

    # Get the input array
    array = greens_function.array.copy()
    if tail_moments is not None:
        array -= grid_in.evaluate_tail(tail_moments, ordering=greens_function.ordering)

    # Perform the Fourier transform
    array = util.einsum("w...,w->w...", array, shifts[0])
    array = fft(array, max(len(grid_in), len(grid_out)), axis=0)[: len(grid_out)]
    array = util.einsum("w...,w->w...", array, shifts[1])

    # Normalise
    array *= norm

    # Add tail
    if tail_moments is not None:
        array += grid_out.evaluate_tail(tail_moments, ordering=greens_function.ordering)

    return greens_function.__class__(
        grid_out,
        array,
        reduction=greens_function.reduction,
        ordering=greens_function.ordering,
        hermitian=greens_function.hermitian,
    )


def fourier_transform_real(
    greens_function: Dynamic[BaseGrid],
    grid: BaseGrid,
    tail_moments: tuple[Array, ...] | None = None,
) -> Dynamic[BaseGrid]:
    """Fourier transform between real frequency and imaginary time grids.

    Args:
        greens_function: Dynamic quantity in real frequency or time domain.
        grid: Target grid for the Fourier transform.
        tail_moments: Moments of the high-frequency tail.

    Returns:
        Dynamic quantity in the target domain.
    """
    grid_in, grid_out = greens_function.grid, grid
    if not are_dual(grid_in, grid_out):
        raise ValueError("the two grids must be dual to each other.")
    if (not grid_in.reality) or (not grid_out.reality):
        raise ValueError("only real frequency and real time grids are supported.")
    if greens_function.component != Component.FULL:
        raise ValueError("only full component is supported.")
    forward = grid_in.domain == "time"

    # Setup based on direction
    if forward:
        freqs, times = grid_out.points, grid_in.points
        norm = grid_in.separation
        fft = np.fft.fft
    else:
        freqs, times = grid_in.points, grid_out.points
        norm = len(freqs) * grid_in.separation / (2.0 * np.pi)
        fft = np.fft.ifft

    # Get the input array
    array = greens_function.array.copy()
    if tail_moments is not None:
        array -= grid_in.evaluate_tail(tail_moments, ordering=greens_function.ordering)

    # Perform the Fourier transform
    array = np.fft.ifftshift(array, axes=0)
    array = fft(array, axis=0)
    array = np.fft.fftshift(array, axes=0)

    # Normalise
    array *= norm

    # Add tail
    if tail_moments is not None:
        array += grid_out.evaluate_tail(tail_moments, ordering=greens_function.ordering)

    return greens_function.__class__(
        grid_out,
        array,
        reduction=greens_function.reduction,
        ordering=greens_function.ordering,
        hermitian=greens_function.hermitian,
    )

