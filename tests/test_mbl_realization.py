"""Tests for input validation and order reduction in the moment block Lanczos solvers."""

from __future__ import annotations

import numpy as np
import pytest

from dyson.representations.lehmann import Lehmann
from dyson.solvers import MBLGF, MBLSE


def self_energy(nphys: int = 4, naux: int = 40, seed: int = 12345) -> tuple[np.ndarray, Lehmann]:
    """Build a random Hermitian self-energy and static part."""
    rng = np.random.default_rng(seed)
    static = rng.normal(size=(nphys, nphys))
    return static + static.T, Lehmann(rng.normal(size=naux), rng.normal(size=(nphys, naux)))


def solver(max_cycle: int = 2) -> MBLSE:
    """Build an MBLSE solver over realizable moments."""
    static, se = self_energy()
    return MBLSE(static, se.moments(range(2 * max_cycle + 2)), max_cycle=max_cycle)


class TestValidation:
    """The recurrence refuses input that cannot describe a measure."""

    def test_valid_moments_pass(self) -> None:
        """Moments of an actual Lehmann representation are accepted."""
        solver().validate_moments()

    @pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
    def test_non_finite_moments_rejected(self, bad: float) -> None:
        """A non-finite element is caught before any work is done."""
        s = solver()
        s._moments = s.moments.copy()
        s._moments[1, 0, 0] = bad

        with pytest.raises(ValueError, match="non-finite"):
            s.validate_moments()

    def test_non_hermitian_moments_rejected(self) -> None:
        """Moments declared Hermitian must actually be Hermitian."""
        s = solver()
        s._moments = s.moments.copy()
        s._moments[1, 0, 1] += 1.0  # break the symmetry of one element

        with pytest.raises(ValueError, match="not Hermitian"):
            s.validate_moments()

    def test_non_hermitian_solver_accepts_asymmetry(self) -> None:
        """The check applies only when the solver has been told the moments are Hermitian."""
        static, se = self_energy()
        moments = se.moments(range(6)).copy()
        moments[1, 0, 1] += 1.0
        s = MBLSE(static, moments, max_cycle=2, hermitian=False)

        s.validate_moments()

    def test_negative_zeroth_moment_rejected(self) -> None:
        """A negative direction in the zeroth moment is negative spectral weight."""
        s = solver()
        s._moments = s.moments.copy()
        s._moments[0] = np.diag([1.0, 1.0, 1.0, -0.5])

        with pytest.raises(ValueError, match="not positive semi-definite"):
            s.validate_moments()

    def test_roundoff_negative_zeroth_moment_accepted(self) -> None:
        """A negative eigenvalue at rounding scale is not treated as negative weight."""
        s = solver()
        s._moments = s.moments.copy()
        s._moments[0] = np.diag([1.0, 1.0, 1.0, -1e-14])

        s.validate_moments()

    def test_kernel_validates(self) -> None:
        """Validation runs as part of the kernel, not only when called directly."""
        s = solver()
        s._moments = s.moments.copy()
        s._moments[2, 1, 1] = np.nan

        with pytest.raises(ValueError, match="non-finite"):
            s.kernel()


class TestOrderReduction:
    """An order the moments cannot support is reduced, not silently accepted."""

    @staticmethod
    def unrealizable_at_order_one() -> MBLSE:
        """Build a one-dimensional problem whose order-1 variance is negative.

        The zeroth moment is positive, so the input passes validation, but ``m2 < m1**2`` means
        no measure has these moments and the block the first iteration would add has no real
        square root.
        """
        moments = np.array([[[1.0]], [[0.0]], [[-1.0]], [[0.0]]])
        return MBLSE(np.array([[0.0]]), moments, max_cycle=1, calculate_errors=False)

    def test_steps_down_to_the_supportable_order(self) -> None:
        """The solver reduces to the largest order it can realize and records it."""
        s = self.unrealizable_at_order_one()

        s.kernel()

        assert s.max_cycle == 1
        assert s.max_cycle_achieved == 0
        assert s.nmom_conserved(s.max_cycle_achieved) == 2
        assert s.result is not None

    def test_reduced_order_is_actually_conserved(self) -> None:
        """The moments the solver still claims are reproduced."""
        s = self.unrealizable_at_order_one()
        s.calculate_errors = True
        s.kernel()

        errors = s.moment_errors()

        assert errors.orders == (0, 1)
        assert errors.max_relative_frobenius < 1e-10

    def test_realizable_input_is_not_reduced(self) -> None:
        """A supportable order is left alone."""
        s = solver(max_cycle=3)

        s.kernel()

        assert s.max_cycle_achieved == s.max_cycle == 3

    def test_moment_errors_default_to_the_realized_order(self) -> None:
        """Asking for the errors without an order reports what was realized, not what was asked."""
        s = self.unrealizable_at_order_one()
        s.calculate_errors = True
        s.kernel()

        assert s.moment_errors().orders == (0, 1)

    def test_failure_at_the_first_iteration_still_raises(self) -> None:
        """There is no order below zero to step down to, so this must not be swallowed."""
        moments = np.array([[[1.0]], [[0.0]]])
        s = MBLSE(np.array([[0.0]]), moments, max_cycle=0, calculate_errors=False)
        # Force the zeroth iteration to fail by making the zeroth moment indefinite past the
        # point validation would allow, bypassing validation to reach the recurrence.
        s._moments = np.array([[[-1.0]], [[0.0]]])

        with pytest.raises(ValueError):
            s.recurrence_iteration(0)


class TestMomentUsage:
    """The solver says which of the supplied moments it consumes."""

    def test_all_moments_used_when_the_count_matches(self, capsys) -> None:
        """A matching count reports nothing, because nothing is left over."""
        s = solver(max_cycle=2)

        s.report_moment_usage()

        assert "supplied moments" not in capsys.readouterr().out

    def test_surplus_moments_reported(self) -> None:
        """An odd count always leaves one moment unconstrained; say so rather than lose it."""
        static, se = self_energy()
        s = MBLSE(static, se.moments(range(7)), max_cycle=2)

        assert np.shape(s.moments)[0] == 7
        assert s.nmom_conserved(s.max_cycle) == 6

    def test_inferred_order_floors_an_odd_count(self) -> None:
        """With no explicit max_cycle an odd count infers the order that fits inside it."""
        static, se = self_energy()

        s = MBLSE(static, se.moments(range(7)))

        assert s.max_cycle == 2
        assert s.nmom_conserved(s.max_cycle) == 6


def test_mblgf_validates_and_reduces() -> None:
    """The Green's function solver shares the behaviour, since both live on the base class."""
    _, se = self_energy()
    s = MBLGF(se.moments(range(6)), max_cycle=2)

    s.kernel()

    assert s.max_cycle_achieved == 2

    bad = MBLGF(se.moments(range(6)).copy(), max_cycle=2)
    bad._moments[0] = np.diag([1.0, 1.0, 1.0, -0.5])
    with pytest.raises(ValueError, match="not positive semi-definite"):
        bad.kernel()
