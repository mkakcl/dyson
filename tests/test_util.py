"""Tests for :module:`~dyson.util`."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from dyson import util
from dyson.util import linalg

if TYPE_CHECKING:
    from typing import Callable

    from pyscf import scf

    from dyson.expressions.expression import BaseExpression

    from .conftest import ExactGetter, Helper


def test_moments_conversion(
    helper: Helper,
    mf: scf.hf.RHF,
    expression_cls: type[BaseExpression],
    exact_cache: ExactGetter,
) -> None:
    """Test the conversion of moments between self-energy and Green's function."""
    # Get the quantities required from the expression
    expression = expression_cls.from_mf(mf)
    if expression.nconfig > 1024:
        pytest.skip("Skipping test for large Hamiltonian")

    # Solve the Hamiltonian
    solver = exact_cache(mf, expression_cls)

    assert solver.result is not None
    assert solver.nphys == expression.nphys
    assert solver.hermitian == expression.hermitian_upfolded

    # Get the self-energy and Green's function from the solver
    static = solver.result.get_static_self_energy()
    self_energy = solver.result.get_self_energy()
    greens_function = solver.result.get_greens_function()

    assert self_energy.nphys == expression.nphys
    assert greens_function.nphys == expression.nphys
    assert helper.recovers_greens_function(static, self_energy, greens_function)

    # Get the moments from the self-energy and Green's function
    se_moments = self_energy.moments(range(4))
    gf_moments = greens_function.moments(range(6))

    # Recover the self-energy from the Green's function moments
    static_other, se_moments_other = util.gf_moments_to_se_moments(gf_moments)
    gf_moments_other = util.se_moments_to_gf_moments(static, se_moments, overlap=gf_moments[0])

    assert helper.are_equal_arrays(static, static_other)
    if expression.hermitian_upfolded:
        assert helper.have_equal_moments(se_moments, se_moments_other, 4)
        assert helper.have_equal_moments(gf_moments, gf_moments_other, 6)
    else:
        assert helper.have_equal_moments(se_moments, se_moments_other, 4, tol=5e-7)
        assert helper.have_equal_moments(gf_moments, gf_moments_other, 6, tol=5e-7)


class TestCacheByID:
    """The cache is keyed on the arguments a call resolves to, not on how it was written."""

    @staticmethod
    def _cached() -> tuple[Callable, list[int]]:
        """Build a cached function that records how often it actually ran."""
        calls: list[int] = []

        @util.cache_by_id
        def f(matrix: np.ndarray, scale: float = 2.0, offset: float | None = None) -> np.ndarray:
            calls.append(1)
            return matrix * scale

        return f, calls

    def test_a_stated_default_reuses_the_implicit_entry(self) -> None:
        """Passing a default explicitly must not compute a second, separate entry."""
        f, calls = self._cached()
        matrix = np.eye(3)

        first = f(matrix)
        second = f(matrix, scale=2.0)

        assert second is first
        assert len(calls) == 1

    def test_a_different_value_is_not_reused(self) -> None:
        """Normalising the key must not merge calls that differ."""
        f, calls = self._cached()
        matrix = np.eye(3)

        f(matrix)
        f(matrix, scale=3.0)

        assert len(calls) == 2

    def test_positional_and_keyword_forms_agree(self) -> None:
        """The same argument given either way is the same computation."""
        f, calls = self._cached()
        matrix = np.eye(3)

        assert f(matrix, 3.0) is f(matrix, scale=3.0)
        assert len(calls) == 1

    def test_matrix_power_defaults_are_not_a_separate_entry(self) -> None:
        """The property the moment block Lanczos tolerances rely on, at the real call site."""
        matrix = np.diag([4.0, 1.0])

        implicit = util.matrix_power_with_info(matrix, 0.5)
        explicit = util.matrix_power_with_info(
            matrix, 0.5, atol=linalg.MATRIX_POWER_ATOL, rtol=linalg.MATRIX_POWER_RTOL
        )

        assert explicit is implicit
