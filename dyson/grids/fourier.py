"""Fourier transformation."""

from __future__ import annotations

from typing import TYPE_CHECKING
import warnings

from dyson import util
from dyson import numpy as np
from dyson.representations.enums import Component

if TYPE_CHECKING:
    from dyson.representations.dynamic import Dynamic
    from dyson.grids.grid import BaseGrid
    from dyson.grids.frequency import GridIF
    from dyson.grids.time import GridIT
    from dyson.typing import Array


#def _apply_imag_factor(
#    array: Array, grid_it: GridIT, grid_if: GridIF, inverse: bool = False
#) -> Array:
#    """Apply the exponential factor for Fourier transform on imaginary axes."""
#    factor = np.exp(1j * np.pi * grid_it.points / grid_if.beta) * grid_if.beta
#    if inverse:
#        factor = 1 / factor
#    return util.einsum("w...,w->w...", array, factor)
#
#
#def fourier_transform_imag(
#    greens_function_it: Dynamic[GridIT],
#    grid_if: GridIF,
#    tail_moments: tuple[Array, ...] | None = None,
#) -> Dynamic[GridIF]:
#    """Forward Fourier transform from imaginary time to imaginary frequency domain.
#
#    Args:
#        gf_if: Dynamic quantity in imaginary time domain.
#        tail_moments: Moments of the high-frequency tail.
#
#    Returns:
#        Dynamic quantity in imaginary frequency domain.
#    """
#    grid_it = greens_function_it.grid
#    if not np.isclose(grid_it.beta, grid_if.beta):
#        raise ValueError("the beta of the two grids must be the same.")
#    if not (grid_it.uniformly_spaced and grid_if.uniformly_spaced):
#        raise ValueError("only uniform grids are supported.")
#    if not (grid_it.uniformly_weighted and grid_if.uniformly_weighted):
#        raise ValueError("only uniform weights are supported.")
#    if greens_function_it.component != Component.FULL:
#        raise ValueError("only FFT for full component is supported.")
#    #if tail_moments is None:
#    #    tail_moments = (greens_function_it.reduction.identity(greens_function_it.nphys),)
#
#    # Get the array
#    array_it = greens_function_it.array.copy()
#
#    if tail_moments is not None:
#        # Subtract tail (treated analytically)
#        array_it -= grid_it.evaluate_tail(tail_moments, ordering=greens_function_it.ordering)
#
#    # Perform FFT
#    array_it = _apply_imag_factor(array_it, grid_it, grid_if, inverse=False)
#    array_if = np.fft.fft(array_it, len(grid_if), axis=0)
#
#    if tail_moments is not None:
#        # Add analytic tail
#        array_if += grid_if.evaluate_tail(tail_moments, ordering=greens_function_it.ordering)
#
#    # Include normalisation from grid weights
#    array_if *= np.sum(grid_if.weights) / np.sum(grid_it.weights)
#
#    return greens_function_it.__class__(
#        grid_if,
#        array_if,
#        reduction=greens_function_it.reduction,
#        ordering=greens_function_it.ordering,
#        hermitian=greens_function_it.hermitian,
#    )
#
#
#def inverse_fourier_transform_imag(
#    greens_function_if: Dynamic[GridIF],
#    grid_it: GridIT,
#    tail_moments: tuple[Array, ...] | None = None,
#) -> Dynamic[GridIT]:
#    """Inverse Fourier transform from imaginary frequency to imaginary time domain.
#
#    Args:
#        gf_if: Dynamic quantity in imaginary frequency domain.
#        tail_moments: Moments of the high-frequency tail.
#
#    Returns:
#        Dynamic quantity in imaginary time domain.
#    """
#    grid_if = greens_function_if.grid
#    if not np.isclose(grid_if.beta, grid_it.beta):
#        raise ValueError("the beta of the two grids must be the same.")
#    if greens_function_if.component != Component.FULL:
#        raise ValueError("only IFFT for full component is supported.")
#    if not (grid_it.uniformly_spaced and grid_if.uniformly_spaced):
#        raise ValueError("only uniform grids are supported.")
#    if not (grid_it.uniformly_weighted and grid_if.uniformly_weighted):
#        raise ValueError("only uniform weights are supported.")
#    #if tail_moments is None:
#    #    tail_moments = (greens_function_if.reduction.identity(greens_function_if.nphys),)
#
#    # Get the array
#    array_if = greens_function_if.array.copy()
#
#    if tail_moments is not None:
#        # Subtract tail (treated analytically)
#        array_if -= grid_if.evaluate_tail(tail_moments, ordering=greens_function_if.ordering)
#
#    # Perform IFFT
#    array_it = np.fft.ifft(array_if, len(grid_it), axis=0)
#    array_it = _apply_imag_factor(array_it, grid_it, grid_if, inverse=True)
#
#    if tail_moments is not None:
#        # Add analytic tail
#        array_it += grid_it.evaluate_tail(tail_moments, ordering=greens_function_if.ordering)
#
#    # Include normalisation from grid weights
#    array_it *= np.sum(grid_it.weights) / np.sum(grid_if.weights)
#
#    return greens_function_if.__class__(
#        grid_it,
#        array_it,
#        reduction=greens_function_if.reduction,
#        ordering=greens_function_if.ordering,
#        hermitian=greens_function_if.hermitian,
#    )
#
#
#def _shift_it(array: Array, grid_if: GridIF, grid_it: GridIT, inverse: bool = False) -> Array:
#    """Shift the array for Fourier transform on imaginary time grid."""
#    shift = np.exp(1.0j * np.pi * grid_it.points / grid_it.beta)
#    if inverse:
#        shift = 1 / shift
#    return util.einsum("w...,w->w...", array, shift)
#
#
#def _shift_if(array: Array, grid_if: GridIF, grid_it: GridIT, inverse: bool = False) -> Array:
#    """Shift the array for Fourier transform on imaginary frequency grid."""
#    shift = np.exp(1.0j * grid_it.points[0] * (grid_if.points - np.pi) / grid_if.beta)
#    if inverse:
#        shift = 1 / shift
#    return util.einsum("w...,w->w...", array, shift)
#
#
#def fourier_transform_imag(
#    greens_function_it: Dynamic[GridIT],
#    grid_if: GridIF,
#    tail_moments: tuple[Array, ...] | None = None,
#) -> Dynamic[GridIF]:
#    """Forward Fourier transform from imaginary time to imaginary frequency domain.
#
#    Args:
#        gf_if: Dynamic quantity in imaginary time domain.
#        tail_moments: Moments of the high-frequency tail.
#
#    Returns:
#        Dynamic quantity in imaginary frequency domain.
#    """
#    grid_it = greens_function_it.grid
#    if not np.isclose(grid_it.beta, grid_if.beta):
#        raise ValueError("the beta of the two grids must be the same.")
#    if not (grid_it.uniformly_spaced and grid_if.uniformly_spaced):
#        raise ValueError("only uniform grids are supported.")
#    if not (grid_it.uniformly_weighted and grid_if.uniformly_weighted):
#        raise ValueError("only uniform weights are supported.")
#    if greens_function_it.component != Component.FULL:
#        raise ValueError("only FFT for full component is supported.")
#
#    # Get the array
#    array_it = greens_function_it.array.copy()
#
#    if tail_moments is not None:
#        # Subtract tail (treated analytically)
#        array_it -= grid_it.evaluate_tail(tail_moments, ordering=greens_function_it.ordering)
#
#    # Perform FFT
#    array_it = _shift_it(array_it, grid_if, grid_it, inverse=False)
#    array_if = np.fft.ifft(array_it, len(grid_if), axis=0)
#    array_if = _shift_if(array_if, grid_if, grid_it, inverse=False)
#
#    # Normalise
#    array_if *= grid_if.beta
#
#    if tail_moments is not None:
#        # Add analytic tail
#        array_if += grid_if.evaluate_tail(tail_moments, ordering=greens_function_it.ordering)
#
#    return greens_function_it.__class__(
#        grid_if,
#        array_if,
#        reduction=greens_function_it.reduction,
#        ordering=greens_function_it.ordering,
#        hermitian=greens_function_it.hermitian,
#    )
#
#
#def inverse_fourier_transform_imag(
#    greens_function_if: Dynamic[GridIF],
#    grid_it: GridIT,
#    tail_moments: tuple[Array, ...] | None = None,
#) -> Dynamic[GridIT]:
#    """Inverse Fourier transform from imaginary frequency to imaginary time domain.
#
#    Args:
#        gf_if: Dynamic quantity in imaginary frequency domain.
#        tail_moments: Moments of the high-frequency tail.
#
#    Returns:
#        Dynamic quantity in imaginary time domain.
#    """
#    grid_if = greens_function_if.grid
#    if not np.isclose(grid_if.beta, grid_it.beta):
#        raise ValueError("the beta of the two grids must be the same.")
#    if greens_function_if.component != Component.FULL:
#        raise ValueError("only IFFT for full component is supported.")
#    if not (grid_it.uniformly_spaced and grid_if.uniformly_spaced):
#        raise ValueError("only uniform grids are supported.")
#    if not (grid_it.uniformly_weighted and grid_if.uniformly_weighted):
#        raise ValueError("only uniform weights are supported.")
#
#    # Get the array
#    array_if = greens_function_if.array.copy()
#
#    if tail_moments is not None:
#        # Subtract tail (treated analytically)
#        array_if -= grid_if.evaluate_tail(tail_moments, ordering=greens_function_if.ordering)
#
#    # Perform IFFT
#    array_if = _shift_if(array_if, grid_if, grid_it, inverse=True)
#    array_it = np.fft.fft(array_if, len(grid_it), axis=0)
#    array_it = _shift_it(array_it, grid_if, grid_it, inverse=True)
#
#    # Normalise
#    array_it /= grid_if.beta
#
#    if tail_moments is not None:
#        # Add analytic tail
#        array_it += grid_it.evaluate_tail(tail_moments, ordering=greens_function_if.ordering)
#
#    return greens_function_if.__class__(
#        grid_it,
#        array_it,
#        reduction=greens_function_if.reduction,
#        ordering=greens_function_if.ordering,
#        hermitian=greens_function_if.hermitian,
#    )


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
    if grid_in.reality or grid_out.reality or (
        {grid_in.domain, grid_out.domain} != {"frequency", "time"}
    ):
        raise ValueError("only imaginary frequency and imaginary time grids is supported.")
    if greens_function.component != Component.FULL:
        raise ValueError("only full component is supported.")
    if not (grid_in.uniformly_spaced and grid_out.uniformly_spaced):
        raise ValueError("only uniform grids are supported.")
    if not (grid_in.uniformly_weighted and grid_out.uniformly_weighted):
        raise ValueError("only uniform weights are supported.")
    if not np.isclose(grid_in.beta, grid_out.beta):
        raise ValueError("the beta of the two grids must be the same.")
    forward = grid_in.domain == "frequency"

    # Setup based on direction
    beta = grid_in.beta
    if forward:
        freqs, times = grid_in.points, grid_out.points
        sign = -1
        norm = 1 / beta
        fft = np.fft.fft
    else:
        freqs, times = grid_out.points, grid_in.points
        sign = 1
        norm = beta
        fft = np.fft.ifft

    # Check the grid sizes
    if len(times) < len(freqs) * 2:
        warnings.warn(
            "Consider using at least twice as many time points as frequency points.",
            RuntimeWarning,
            stacklevel=2,
        )

    # Get the shifts
    shift_if = np.exp(1.0j * sign * np.pi * times / beta)
    shift_it = np.exp(1.0j * sign * (freqs - np.pi / beta) * times[0])
    shifts = (shift_it, shift_if) if forward else (shift_if, shift_it)

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
    damping: float | None = None,
) -> Dynamic[BaseGrid]:
    """Fourier transform between real frequency and real time grids.

    Args:
        greens_function: Dynamic quantity in real frequency or time domain.
        grid: Target grid for the Fourier transform.
        damping: Optional exponential damping factor applied in time domain
            as exp(-damping * |t|) when transforming time -> frequency.

    Returns:
        Dynamic quantity in the target domain.
    """
    grid_in, grid_out = greens_function.grid, grid
    if (not grid_in.reality) or (not grid_out.reality) or (
        {grid_in.domain, grid_out.domain} != {"frequency", "time"}
    ):
        raise ValueError("only real frequency and real time grids are supported.")
    if greens_function.component != Component.FULL:
        raise ValueError("only full component is supported.")
    if not (grid_in.uniformly_spaced and grid_out.uniformly_spaced):
        raise ValueError("only uniform grids are supported.")
    if not (grid_in.uniformly_weighted and grid_out.uniformly_weighted):
        raise ValueError("only uniform weights are supported.")

    forward = grid_in.domain == "time"  # time -> frequency

    time_grid = grid_in if forward else grid_out
    freq_grid = grid_out if forward else grid_in

    times = time_grid.points
    freqs = freq_grid.points

    dt = float(times[1] - times[0])
    if not np.allclose(np.diff(times), dt):
        raise ValueError("time grid must be uniformly spaced.")
    if dt <= 0:
        raise ValueError("time grid must be increasing.")

    n_in = len(grid_in)
    n_out = len(grid_out)
    n_fft = max(n_in, n_out)

    # FFT-compatible frequency grid (ascending) associated with this dt.
    omega_fft = np.fft.fftshift(np.fft.fftfreq(n_fft, d=dt) * (2.0 * np.pi))
    domega = float(omega_fft[1] - omega_fft[0])

    # Basic sanity: require requested frequency spacing to match FFT spacing.
    if len(freqs) > 1:
        domega_req = float(freqs[1] - freqs[0])
        if not np.allclose(np.diff(freqs), domega_req):
            raise ValueError("frequency grid must be uniformly spaced.")
        if not np.isclose(domega_req, domega, rtol=1e-7, atol=1e-12):
            raise ValueError(
                "frequency spacing is not FFT-compatible with time spacing: "
                f"dω={domega_req} vs 2π/(N dt)={domega}."
            )

    # Optional warning if ranges look inconsistent.
    if len(freqs) < len(times) // 2 and forward:
        warnings.warn(
            "Consider using comparable numbers of time and frequency points for FFT-based transforms.",
            RuntimeWarning,
            stacklevel=2,
        )

    # Handle a constant time-origin shift: times need not be centred at 0.
    # We assume times = t0 + (j - N/2) dt after fftshift; any constant shift is corrected.
    idx = np.arange(n_fft)
    t_expected = (idx - n_fft // 2) * dt
    t_shift = float(times[0] - t_expected[0])

    array = greens_function.array

    if forward:
        # time -> frequency: G(ω) ≈ dt * Σ_j e^{+i ω t_j} G(t_j)
        # Implemented as: dt * N * FFTshift( IFFT( IFFTshift(G(t)) ) ) with phase for t_shift.
        work = np.zeros((n_fft, *array.shape[1:]), dtype=complex)
        work[: n_in] = array

        if damping is not None and damping > 0:
            t_full = t_expected + t_shift
            work *= np.exp(-damping * np.abs(t_full))[:, None, None]

        work = np.fft.ifftshift(work, axes=0)
        work = np.fft.ifft(work, axis=0)
        work = np.fft.fftshift(work, axes=0)

        omega_full = omega_fft
        work *= (dt * n_fft) * np.exp(1.0j * omega_full * t_shift)[:, None, None]

        out = work[: n_out]

    else:
        # frequency -> time: G(t) ≈ (dω/2π) * Σ_k e^{-i ω t} G(ω)
        # Implemented as: (dω/2π) * FFTshift( FFT( IFFTshift(G(ω)) ) ) with inverse phase.
        work = np.zeros((n_fft, *array.shape[1:]), dtype=complex)
        work[: n_in] = array

        work = np.fft.ifftshift(work, axes=0)
        work = np.fft.fft(work, axis=0)
        work = np.fft.fftshift(work, axes=0)

        t_full = t_expected + t_shift
        work *= (domega / (2.0 * np.pi)) * np.exp(-1.0j * omega_fft * t_shift)[:, None, None]

        out = work[: n_out]

    return greens_function.__class__(
        grid_out,
        out,
        reduction=greens_function.reduction,
        ordering=greens_function.ordering,
        hermitian=greens_function.hermitian,
    )
