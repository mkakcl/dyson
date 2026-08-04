"""Tests for the matrix-power tolerances exposed by the moment block Lanczos solvers."""

from __future__ import annotations

import contextlib
import io
import warnings
from typing import TYPE_CHECKING

import numpy as np
import pytest

from dyson import util
from dyson.representations.lehmann import Lehmann
from dyson.solvers import MBLGF, MBLSE
from dyson.util.linalg import MATRIX_POWER_ATOL, MATRIX_POWER_NEG_RTOL, MATRIX_POWER_RTOL

if TYPE_CHECKING:
    from typing import Any

    from dyson.typing import Array

TOLERANCES = ("atol", "rtol", "neg_atol", "neg_rtol")


def self_energy(nphys: int = 4, naux: int = 40, seed: int = 12345) -> tuple[Array, Lehmann]:
    """Build a random Hermitian self-energy and static part."""
    rng = np.random.default_rng(seed)
    static = rng.normal(size=(nphys, nphys))
    return static + static.T, Lehmann(rng.normal(size=naux), rng.normal(size=(nphys, naux)))


def solver(max_cycle: int = 2, **kwargs: Any) -> MBLSE:
    """Build an MBLSE solver over realizable moments."""
    static, se = self_energy()
    return MBLSE(static, se.moments(range(2 * max_cycle + 2)), max_cycle=max_cycle, **kwargs)


def run(s: MBLSE | MBLGF) -> None:
    """Run a solver without its output."""
    with contextlib.redirect_stdout(io.StringIO()):
        s.kernel()


class TestOptionSurface:
    """The tolerances are options like any other, and only the ones set are forwarded."""

    @pytest.mark.parametrize("name", TOLERANCES)
    def test_accepted_by_the_constructor(self, name: str) -> None:
        """Each tolerance can be given at construction."""
        assert getattr(solver(**{name: 1e-7}), name) == 1e-7

    @pytest.mark.parametrize("name", TOLERANCES)
    def test_accepted_by_set_options(self, name: str) -> None:
        """Each tolerance can be set after construction."""
        s = solver()

        s.set_options(**{name: 1e-7})

        assert getattr(s, name) == 1e-7

    def test_accepted_by_mblgf(self) -> None:
        """The Green's function solver shares the option surface, not only MBLSE."""
        _, se = self_energy()

        s = MBLGF(se.moments(range(6)), max_cycle=2, atol=1e-7, neg_rtol=1e-6)

        assert s.matrix_power_options == {"atol": 1e-7, "neg_rtol": 1e-6}

    def test_unknown_option_still_raises(self) -> None:
        """Widening the surface must not turn it into an anything-goes surface."""
        with pytest.raises(ValueError, match="Unknown option"):
            solver(support_tol=1e-7)

    def test_unset_solver_forwards_nothing(self) -> None:
        """An unset solver defers to the library rather than restating its defaults."""
        assert solver().matrix_power_options == {}

    def test_only_what_was_set_is_forwarded(self) -> None:
        """Setting one tolerance does not pin the other three to a copy of their defaults."""
        assert solver(rtol=1e-9).matrix_power_options == {"rtol": 1e-9}


class TestForwarding:
    """The tolerances reach every matrix power the recurrence takes."""

    @staticmethod
    def _spy(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
        """Record the keyword arguments of every ``matrix_power`` call."""
        seen: list[dict[str, Any]] = []
        real = util.matrix_power

        def spy(matrix: Array, power: float, **kwargs: Any) -> Any:
            seen.append(kwargs)
            return real(matrix, power, **kwargs)

        monkeypatch.setattr(util, "matrix_power", spy)
        return seen

    def test_every_call_carries_the_tolerances(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Not one of the powers in the recurrence is left on the library defaults."""
        seen = self._spy(monkeypatch)
        options = {"atol": 1e-11, "rtol": 1e-13, "neg_atol": 1e-11, "neg_rtol": 1e-9}

        run(solver(**options))

        assert seen, "the recurrence took no matrix powers, so nothing was tested"
        for call in seen:
            assert {name: call.get(name) for name in TOLERANCES} == options

    def test_mblgf_calls_carry_them_too(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The same holds for the Green's function recurrence."""
        seen = self._spy(monkeypatch)
        _, se = self_energy()

        run(MBLGF(se.moments(range(6)), max_cycle=2, atol=1e-11))

        assert seen
        assert all(call.get("atol") == 1e-11 for call in seen)

    def test_unset_calls_pass_no_tolerance_at_all(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A solver that sets nothing calls exactly as it did before these options existed."""
        seen = self._spy(monkeypatch)

        run(solver())

        assert seen
        assert all(not (set(call) & set(TOLERANCES)) for call in seen)

    def test_stating_the_defaults_changes_nothing(self) -> None:
        """The library defaults, passed explicitly, reproduce the unset result exactly.

        This is what makes the option surface safe to adopt: a caller that wants its numerical
        policy written down rather than inherited gets the same numbers, bit for bit.
        """
        unset = solver()
        stated = solver(
            atol=MATRIX_POWER_ATOL,
            rtol=MATRIX_POWER_RTOL,
            neg_atol=MATRIX_POWER_ATOL,
            neg_rtol=MATRIX_POWER_NEG_RTOL,
        )

        run(unset)
        run(stated)

        assert unset.result is not None and stated.result is not None
        np.testing.assert_array_equal(unset.result.eigvals, stated.result.eigvals)
        assert unset.moment_error() == stated.moment_error()


class TestEffect:
    """The tolerances are used, not merely accepted and carried."""

    def test_a_loose_support_cutoff_degrades_the_moments(self) -> None:
        """Discarding real support costs moment accuracy, and the option can cause that."""
        tight = solver()
        loose = solver(rtol=0.5)

        run(tight)
        run(loose)

        assert tight.moment_error() < 1e-10
        assert loose.moment_error() > 1e-3

    def test_negativity_tolerance_decides_what_is_roundoff(self) -> None:
        """A direction the default clips as roundoff is material once the tolerance is zero."""
        matrix = np.diag([1.0, 0.5, -1e-9])

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            util.matrix_power(matrix, 0.5, strict=True)

        with pytest.raises(util.NotPositiveSemiDefiniteError):
            util.matrix_power(matrix, 0.5, strict=True, neg_atol=0.0, neg_rtol=0.0)
