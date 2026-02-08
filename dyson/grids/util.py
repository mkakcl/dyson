"""Utility functions for grids."""

from __future__ import annotations

from typing import TYPE_CHECKING

from dyson import numpy as np

if TYPE_CHECKING:
    from dyson.grids.grid import BaseGrid


def are_dual(grid1: BaseGrid, grid2: BaseGrid) -> bool:
    """Check if a pair of grids are dual to each other.

    Dual grids are related by a Fourier transform.

    Args:
        grid1: The first grid.
        grid2: The second grid.

    Returns:
        Whether the grids are dual to each other.
    """
    if not {grid1.domain, grid2.domain} == {"time", "frequency"}:
        raise ValueError("one grid must be in time domain and the other in frequency domain")
    if grid1.reality != grid2.reality:
        raise ValueError("both grids must be either real or imaginary")
    if not grid1.uniformly_spaced or not grid2.uniformly_spaced:
        raise ValueError("duality only supported for uniformly spaced grids")
    if not grid1.uniformly_weighted or not grid2.uniformly_weighted:
        raise ValueError("duality only supported for uniformly weighted grids")
    time, freq = (grid1, grid2) if grid1.domain == "time" else (grid2, grid1)
    time_recov = time.from_dual(freq)
    freq_recov = freq.from_dual(time)
    same_points = len(time) == len(time_recov) and np.allclose(time.points, time_recov.points)
    same_points &= len(freq) == len(freq_recov) and np.allclose(freq.points, freq_recov.points)
    same_eta = np.isclose(getattr(time, "eta", 0.0), getattr(freq, "eta", 0.0))
    return bool(same_points and same_eta)
