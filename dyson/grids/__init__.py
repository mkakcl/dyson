r"""Grids for Green's functions and self-energies.

Grids are arrays of points in either the frequency or time domain.


Submodules
----------

.. autosummary::
    :toctree:

    grid
    frequency
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from dyson.grids.frequency import RealFrequencyGrid, GridRF
from dyson.grids.frequency import ImaginaryFrequencyGrid, GridIF
from dyson.grids.time import RealTimeGrid, GridRT
from dyson.grids.time import ImaginaryTimeGrid, GridIT
from dyson.grids.fourier import fourier_transform_imag, fourier_transform_real
from dyson.grids.pade import analytic_continuation_freq_pade

if TYPE_CHECKING:
    from typing import Any

    from dyson.representations import Dynamic
    from dyson.grids.grid import BaseGrid


def transform(dynamic: Dynamic[BaseGrid], grid: BaseGrid, **kwargs: Any) -> Dynamic[BaseGrid]:
    """Transform a dynamic quantity to a new grid using either FFT or AC.

    Currently available transformations are:

    .. code-block:: bash

                   AC
               ─────────>
        GridRF <───────── GridIF
                   AC
         │ ^               │ ^
         │ │               │ │
    IFFT │ │ FFT      IFFT │ │ FFT
         │ │               │ │
         v │               v │

        GridRT           GridIT

    Args:
        dynamic: The dynamic quantity to transform.
        grid: The grid to transform to.
        **kwargs: Additional keyword arguments passed to the transformation function.

    Returns:
        The transformed dynamic quantity.

    Raises:
        NotImplementedError: If the transformation is not implemented.
    """
    if isinstance(dynamic.grid, GridRF) and isinstance(grid, GridRT):
        return fourier_transform_real(dynamic, grid, **kwargs)
    if isinstance(dynamic.grid, GridRT) and isinstance(grid, GridRF):
        return fourier_transform_real(dynamic, grid, **kwargs)
    if isinstance(dynamic.grid, GridIT) and isinstance(grid, GridIF):
        return fourier_transform_imag(dynamic, grid, **kwargs)
    if isinstance(dynamic.grid, GridIF) and isinstance(grid, GridIT):
        return fourier_transform_imag(dynamic, grid, **kwargs)
    if isinstance(dynamic.grid, GridIF) and isinstance(grid, GridRF):
        return analytic_continuation_freq_pade(dynamic, grid, **kwargs)
    if isinstance(dynamic.grid, GridRF) and isinstance(grid, GridIF):
        return analytic_continuation_freq_pade(dynamic, grid, **kwargs)
    raise NotImplementedError(
        f"transformation between {type(dynamic.grid)} and {type(grid)} not implemented"
    )
