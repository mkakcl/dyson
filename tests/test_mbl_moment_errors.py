"""Tests for the moment-error diagnostics of the moment block Lanczos solvers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from dyson.representations.lehmann import Lehmann
from dyson.solvers import MBLGF, MBLSE
from dyson.solvers.static._mbl import MomentErrors, shift_moments

if TYPE_CHECKING:
    from dyson.typing import Array


def _self_energy(nphys: int = 4, naux: int = 40, seed: int = 12345) -> tuple[Array, Lehmann]:
    """Build a random Hermitian self-energy and static part."""
    rng = np.random.default_rng(seed)
    static = rng.normal(size=(nphys, nphys))
    return static + static.T, Lehmann(rng.normal(size=naux), rng.normal(size=(nphys, naux)))


def _solver(max_cycle: int) -> MBLSE:
    """Build and run an MBLSE solver at a given cycle."""
    static, self_energy = _self_energy()
    moments = self_energy.moments(range(2 * max_cycle + 2))
    solver = MBLSE(static, moments, max_cycle=max_cycle, calculate_errors=False)
    solver.kernel()
    return solver


@pytest.mark.parametrize("max_cycle", [0, 1, 2, 3])
def test_reconstructs_every_conserved_moment(max_cycle: int) -> None:
    """The reconstruction covers every moment the recurrence conserves.

    Regression test for the reconstruction reconstructing ``2 * iteration`` moments while the
    error routine compared against ``2 * iteration + 2``: ``zip`` silently dropped the newest
    two moments, and iteration zero compared none at all.
    """
    solver = _solver(max_cycle)

    predicted = solver.reconstruct_moments(max_cycle)
    reference = solver.reference_moments(max_cycle)

    assert len(predicted) == 2 * max_cycle + 2
    assert len(reference) == 2 * max_cycle + 2
    assert solver.moment_errors(iteration=max_cycle).orders == tuple(range(2 * max_cycle + 2))


def test_iteration_zero_compares_moments() -> None:
    """Iteration zero checks the two moments it conserves rather than none."""
    errors = _solver(0).moment_errors(iteration=0)

    assert errors.orders == (0, 1)
    assert len(errors.relative_frobenius) == 2


@pytest.mark.parametrize("max_cycle", [0, 1, 2, 3])
def test_conserved_moments_are_accurate(max_cycle: int) -> None:
    """Every conserved moment is reproduced to near machine precision."""
    errors = _solver(max_cycle).moment_errors(iteration=max_cycle)

    assert errors.max_relative_frobenius < 1e-10
    assert errors.max_relative_max < 1e-10


def test_error_in_highest_moment_is_reported() -> None:
    """A discrepancy in the highest conserved moment reaches the reported error.

    Under the previous ``zip``-truncated comparison this moment was dropped, so a corrupted
    reference reported an error at the level of machine precision.
    """
    solver = _solver(1)
    solver._moments = solver.moments.copy()
    solver._moments[3] += 0.5 * np.max(np.abs(solver._moments[3]))

    errors = solver.moment_errors(iteration=1)

    assert errors.orders == (0, 1, 2, 3)
    assert errors.relative_frobenius[3] > 1e-2
    assert max(errors.relative_frobenius[:3]) < 1e-10
    assert errors.total > 1e-2


def test_mismatched_moment_counts_raise() -> None:
    """Comparing unequal numbers of moments fails rather than truncating silently."""
    solver = _solver(1)
    solver.nmom_conserved = staticmethod(lambda iteration: 2 * iteration + 4)

    with pytest.raises(ValueError, match="reference moments"):
        solver.moment_errors(iteration=1)


def test_reports_both_norms_per_order() -> None:
    """Absolute and relative Frobenius and maximum norms are reported for each order."""
    solver = _solver(2)
    errors = solver.moment_errors(iteration=2)

    for field in (
        errors.absolute_frobenius,
        errors.relative_frobenius,
        errors.absolute_max,
        errors.relative_max,
        errors.scaled,
    ):
        assert len(field) == len(errors.orders)
        assert all(np.isfinite(value) for value in field)

    # The Frobenius norm of a matrix bounds its largest entry from above.
    for absolute_frobenius, absolute_max in zip(errors.absolute_frobenius, errors.absolute_max):
        assert absolute_max <= absolute_frobenius * (1.0 + 1e-12)


def test_norms_scale_as_expected() -> None:
    """The reported norms take the values implied by a known perturbation."""
    solver = _solver(1)
    solver._moments = solver.moments.copy()
    perturbation = np.zeros_like(solver.moments[0])
    perturbation[0, 0] = 1.0
    solver._moments[0] = solver.moments[0] + perturbation
    # The perturbed moment is the reference, so it sets the scale of the relative error.
    reference_scale = np.linalg.norm(solver.moments[0])

    errors = solver.moment_errors(iteration=1)

    # A single corrupted entry: both norms see exactly that entry.
    assert errors.absolute_frobenius[0] == pytest.approx(1.0, rel=1e-6)
    assert errors.absolute_max[0] == pytest.approx(1.0, rel=1e-6)
    assert errors.relative_frobenius[0] == pytest.approx(1.0 / reference_scale, rel=1e-6)


def test_moment_error_is_the_aggregate() -> None:
    """The scalar error stays the sum of the per-order scaled errors."""
    solver = _solver(2)
    errors = solver.moment_errors(iteration=2)

    assert solver.moment_error(iteration=2) == pytest.approx(errors.total)
    assert errors.total == pytest.approx(sum(errors.scaled))


def test_mblgf_reconstruction_is_unchanged() -> None:
    """MBLGF, whose count was already correct, is unaffected."""
    _, self_energy = _self_energy()
    solver = MBLGF(self_energy.moments(range(4)), max_cycle=1, calculate_errors=False)
    solver.kernel()

    errors = solver.moment_errors(iteration=1)

    assert errors.orders == (0, 1, 2, 3)
    assert errors.max_relative_frobenius < 1e-10


class TestChempotShift:
    """Tests for separating errors either side of a chemical-potential pole shift."""

    def test_shift_matches_explicitly_shifted_poles(self) -> None:
        """The binomial transform reproduces moments taken about a shifted origin."""
        _, self_energy = _self_energy()
        chempot = 0.37

        shifted = shift_moments(self_energy.moments(range(6)), chempot)
        explicit = Lehmann(
            self_energy.energies - chempot, self_energy.couplings, sort=False
        ).moments(range(6))

        assert np.allclose(shifted, explicit, atol=1e-10)

    def test_zero_shift_is_the_identity(self) -> None:
        """Shifting by zero leaves the moments untouched."""
        _, self_energy = _self_energy()
        moments = self_energy.moments(range(6))

        assert np.allclose(shift_moments(moments, 0.0), moments, atol=1e-14)

    def test_no_shift_reported_when_chempot_is_zero(self) -> None:
        """With no shift in effect, no second set of errors is reported."""
        errors = _solver(1).moment_errors(iteration=1)

        assert errors.chempot == 0.0
        assert errors.shifted is None

    def test_shifted_errors_reported_when_chempot_is_set(self) -> None:
        """A non-zero chemical potential adds errors for the shifted poles.

        :meth:`Lehmann.moments` always measures poles from the origin, so a chemical potential
        is metadata that the moments themselves ignore. Reporting both sets keeps a shift
        visible as a change of convention instead of an unexplained change in the error.
        """
        solver = _solver(1)
        chempot = 0.37
        unshifted = solver.reconstruct_representation(1)
        solver.reconstruct_representation = lambda iteration: Lehmann(  # type: ignore[method-assign]
            unshifted.energies, unshifted.couplings, chempot=chempot, sort=False
        )

        errors = solver.moment_errors(iteration=1)

        assert errors.chempot == pytest.approx(chempot)
        assert isinstance(errors.shifted, MomentErrors)
        assert errors.shifted.orders == errors.orders
        # The reconstruction is still exact, so shifting both sides keeps it exact.
        assert errors.shifted.max_relative_frobenius < 1e-10
