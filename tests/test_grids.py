"""Tests for :module:`~dyson.grids`."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from dyson.grids import GridRF, GridRT, GridIF, GridIT, transform
from dyson.grids.grid import BaseGrid
from dyson.representations.spectral import Spectral
from dyson.representations.enums import Ordering, Reduction, Component

if TYPE_CHECKING:
    from pyscf import scf

    from dyson.expressions.expression import BaseExpression, ExpressionCollection
    from dyson.representations.dynamic import Dynamic

    from .conftest import ExactGetter, Helper


def with_dual(grid: BaseGrid) -> tuple[BaseGrid, BaseGrid]:
    """Return a grid and its dual."""
    dual_class = {
        GridIF: GridIT,
        GridIT: GridIF,
        GridRF: GridRT,
        GridRT: GridRF,
    }[type(grid)]
    return grid, dual_class.from_dual(grid)


def get_dynamic_pair(
    helper: Helper,
    mf: scf.hf.RHF,
    expression_method: ExpressionCollection,
    exact_cache: ExactGetter,
    ordering: Ordering,
    reduction: Reduction,
    component: Component,
    grid1: BaseGrid,
    grid2: BaseGrid,
) -> tuple[Dynamic, Dynamic]:
    """Get the pair of dynamic Green's functions for the given parameters."""
    expression_h = expression_method.h.from_mf(mf)
    expression_p = expression_method.p.from_mf(mf)
    if expression_h.nconfig > 1024 or expression_p.nconfig > 1024:
        pytest.skip("Skipping test for large Hamiltonian")

    # Solve the Hamiltonian exactly
    exact_h = exact_cache(mf, expression_method.h)
    exact_p = exact_cache(mf, expression_method.p)
    assert exact_h.result is not None
    assert exact_p.result is not None
    result = Spectral.combine_for_greens_function(exact_h.result, exact_p.result)
    gf1 = grid1.evaluate_lehmann(
        result.get_greens_function(),
        ordering=ordering,
        reduction=reduction,
        component=component,
    )
    gf2 = grid2.evaluate_lehmann(
        result.get_greens_function(),
        ordering=ordering,
        reduction=reduction,
        component=component,
    )

    return gf1, gf2


@pytest.mark.parametrize("ordering", [Ordering.RETARDED, Ordering.ADVANCED])
@pytest.mark.parametrize("reduction", [Reduction.NONE, Reduction.DIAG])
@pytest.mark.parametrize("component", [Component.FULL])
@pytest.mark.parametrize("grid_rf", [GridRF.from_uniform(-8, 8, 501, eta=0.1)])
@pytest.mark.parametrize("grid_if", [GridIF.from_uniform(501, beta=32)])
def test_transform_rf_if(
    helper: Helper,
    mf: scf.hf.RHF,
    expression_method: ExpressionCollection,
    exact_cache: ExactGetter,
    ordering: Ordering,
    reduction: Reduction,
    component: Component,
    grid_rf: BaseGrid,
    grid_if: BaseGrid,
) -> None:
    """Test Fourier transform between imaginary time and frequency."""
    gf_rf, gf_if = get_dynamic_pair(
        helper,
        mf,
        expression_method,
        exact_cache,
        ordering,
        reduction,
        component,
        grid_rf,
        grid_if,
    )

    # Transform the Green's functions
    gf_if_recov = transform(gf_rf, grid_if)
    gf_rf_recov = transform(gf_if, grid_rf)

    # Check that the recovered Green's functions match the original ones
    assert np.mean(np.abs(gf_rf_recov - gf_rf)) < 1e-1  # approximate
    assert np.allclose(gf_if_recov, gf_if)  # exact


@pytest.mark.parametrize("ordering", [Ordering.RETARDED, Ordering.ADVANCED, Ordering.ORDERED])
@pytest.mark.parametrize("reduction", [Reduction.NONE, Reduction.DIAG])
@pytest.mark.parametrize("component", [Component.FULL])
@pytest.mark.parametrize("grid_if, grid_it", [with_dual(GridRF.from_uniform(-8, 8, 256, eta=0.25))])
def test_transform_if_it(
    helper: Helper,
    mf: scf.hf.RHF,
    expression_method: ExpressionCollection,
    exact_cache: ExactGetter,
    ordering: Ordering,
    reduction: Reduction,
    component: Component,
    grid_if: BaseGrid,
    grid_it: BaseGrid,
) -> None:
    """Test Fourier transform between imaginary time and frequency."""
    gf_if, gf_it = get_dynamic_pair(
        helper,
        mf,
        expression_method,
        exact_cache,
        ordering,
        reduction,
        component,
        grid_if,
        grid_it,
    )

    # Transform the Green's functions
    gf_it_recov = transform(gf_if, grid_it)
    gf_if_recov = transform(gf_it, grid_if)

    # Check that the recovered Green's functions match the original ones
    low_freq = np.abs(grid_if.points) < 2.0
    if not np.allclose(gf_it_recov.array[low_freq], gf_it.array[low_freq]):
        from dyson.plotting import plot_dynamic
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        plot_dynamic(gf_if.copy(reduction="trace", component="real"), ax=ax, fmt="C0.-", label="Original RF")
        plot_dynamic(gf_if_recov.copy(reduction="trace", component="real"), ax=ax, fmt="C1.--", label="Recovered RF")
        plt.xlabel("Frequency")
        plt.ylabel("Re G")
        plt.legend()
        plt.show()
    assert np.allclose(gf_if.array[low_freq], gf_if_recov.array[low_freq])
    #assert np.mean(np.abs(gf_if_recov - gf_if)) < 1e-1
    #assert np.mean(np.abs(gf_it_recov - gf_it)) < 1e-1
