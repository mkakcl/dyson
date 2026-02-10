"""Tests for :module:`~dyson.solvers.dynamic.direct`."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from dyson import util
from dyson.grids import RealFrequencyGrid, ImaginaryFrequencyGrid
from dyson.solvers import Direct, Exact
from dyson.representations.dynamic import Dynamic
from dyson.representations.enums import Ordering
from dyson.representations.spectral import Spectral
from dyson.expressions.hamiltonian import Hamiltonian
from dyson.solvers.recipes import greens_function_from_hamiltonian

if TYPE_CHECKING:
    from typing import Any

    from pyscf import scf

    from dyson.typing import Array
    from dyson.expressions.expression import BaseExpression
    from dyson.grids.grid import BaseGrid

    from .conftest import ExactGetter, ExpressionCollection, Helper


@pytest.mark.parametrize(
    "grid",
    [
        RealFrequencyGrid.from_uniform(-5, 5, 1001, eta=0.2),
        ImaginaryFrequencyGrid.from_uniform(501, beta=32),
    ],
)
@pytest.mark.parametrize("ordering", [Ordering.ADVANCED, Ordering.RETARDED])
def test_vs_exact_solver(
    helper: Helper,
    mf: scf.hf.RHF,
    expression_method: ExpressionCollection,
    exact_cache: ExactGetter,
    grid: BaseGrid,
    ordering: Ordering,
) -> None:
    """Test frequency-space solver compared to the exact solver."""
    # Get the quantities required from the expressions
    expression_h = expression_method.h.from_mf(mf)
    expression_p = expression_method.p.from_mf(mf)
    if expression_h.nconfig > 1024 or expression_p.nconfig > 1024:
        pytest.skip("Skipping test for large Hamiltonian")

    # Solve the Hamiltonian exactly
    exact_h = exact_cache(mf, expression_method.h)
    exact_p = exact_cache(mf, expression_method.p)
    assert exact_h.result is not None
    assert exact_p.result is not None
    result = Spectral.combine_for_self_energy(exact_h.result, exact_p.result)

    # Get the central self-energy
    overlap = result.get_overlap()
    static = result.get_static_self_energy()
    self_energy = result.get_self_energy()

    # Solve the self-energy exactly to get the exact Green's function
    exact = Exact.from_self_energy(static, self_energy, overlap=overlap)
    exact.kernel()
    se_exact = grid.evaluate_lehmann(exact.result.get_self_energy(), ordering=ordering)
    gf_exact = grid.evaluate_lehmann(exact.result.get_greens_function(), ordering=ordering)

    # Get G_0
    gf_0 = grid.evaluate_lehmann(
        greens_function_from_hamiltonian(static, overlap=overlap, hermitian=result.hermitian),
        ordering=ordering,
    )

    # Solve the Hamiltonian with RealFrequency
    direct = Direct(gf_0, se_exact, overlap=overlap)
    gf = direct.kernel()

    assert np.allclose(gf.array, gf_exact.array)
