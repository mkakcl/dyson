"""Tests for reporting poles that carry no resolvable spectral weight.

A realization emits a fixed number of poles. Where the moments support fewer independent
directions than that, the surplus arrive with couplings at the level of roundoff and
energies taken from a numerically null block -- so their placement is not determined by the
data. These tests cover detecting them, and the growth with moment order that makes an
undetermined energy matter.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from dyson.representations.lehmann import Lehmann

if TYPE_CHECKING:
    from dyson.typing import Array


def build(
    nphys: int = 4,
    naux: int = 20,
    *,
    n_weightless: int = 0,
    weightless_scale: float = 1e-11,
    energies: Array | None = None,
    seed: int = 271828,
) -> Lehmann:
    """Build a Lehmann representation with a controlled number of weightless poles."""
    rng = np.random.default_rng(seed)
    if energies is None:
        energies = rng.normal(size=naux)
    couplings = rng.normal(size=(nphys, naux))
    if n_weightless:
        couplings[:, :n_weightless] *= weightless_scale
    return Lehmann(np.asarray(energies), couplings, sort=False)


@pytest.mark.parametrize("n_weightless", [0, 1, 5, 12])
def test_counts_them(n_weightless: int) -> None:
    """Every pole scaled down to roundoff is found, and no others."""
    lehmann = build(n_weightless=n_weightless)

    found = lehmann.weightless_poles()

    assert found.count == n_weightless
    assert found.naux == lehmann.naux


def test_none_reported_for_an_ordinary_representation() -> None:
    """A representation whose poles all carry weight reports nothing."""
    found = build().weightless_poles()

    assert found.count == 0
    assert found.energy_min is None
    assert found.energy_max is None
    assert found.energy_spread == 0.0
    assert found.worst_moment_contribution == 0.0


def test_threshold_is_scale_aware() -> None:
    """The threshold follows the largest pole weight, not an absolute number.

    Scaling every coupling scales every weight, so the same poles must be reported: a fixed
    cutoff would silently mean something different for every system.
    """
    small = build(n_weightless=5)
    large = Lehmann(small.energies, small.couplings * 1e6, sort=False)

    assert large.weightless_poles().count == small.weightless_poles().count


def test_energy_range_covers_the_weightless_poles_only() -> None:
    """The reported spread describes where the undetermined poles sit."""
    energies = np.concatenate([np.array([-40.0, 40.0]), np.full(18, 0.5)])
    lehmann = build(naux=20, n_weightless=2, energies=energies)

    found = lehmann.weightless_poles()

    assert found.count == 2
    assert found.energy_min == pytest.approx(-40.0)
    assert found.energy_max == pytest.approx(40.0)
    assert found.energy_spread == pytest.approx(80.0)


def test_contribution_grows_with_moment_order() -> None:
    """A weightless pole placed far out contributes more at higher order.

    This is the mechanism the report exists for: the pole is negligible in the function and
    in the low moments, and `e**n` brings it back at high order.
    """
    energies = np.concatenate([np.array([-50.0]), np.full(19, 0.5)])
    lehmann = build(naux=20, n_weightless=1, weightless_scale=1e-7, energies=energies)

    found = lehmann.weightless_poles(orders=8)

    assert found.count == 1
    assert found.moment_contribution[0] < found.moment_contribution[-1]
    # And strictly increasing once the far-out pole starts to tell.
    tail = found.moment_contribution[2:]
    assert all(b > a for a, b in zip(tail, tail[1:]))


def test_placement_changes_the_high_moments_but_not_the_low_ones() -> None:
    """Moving an undetermined pole is invisible at low order and visible at high order.

    The two representations differ only in where a weightless pole sits, which is exactly
    the freedom the realization has no information to fix.
    """
    base = np.full(20, 0.5)
    near = build(
        naux=20,
        n_weightless=1,
        weightless_scale=1e-7,
        energies=np.concatenate([np.array([0.0]), base[1:]]),
    )
    far = build(
        naux=20,
        n_weightless=1,
        weightless_scale=1e-7,
        energies=np.concatenate([np.array([-50.0]), base[1:]]),
    )

    a = near.moments(range(8))
    b = far.moments(range(8))
    rel = [
        float(np.linalg.norm(x - y)) / max(float(np.linalg.norm(x)), 1e-300) for x, y in zip(a, b)
    ]

    assert rel[0] < 1e-12
    assert rel[-1] > rel[0]


def test_weights_per_pole_sums_to_the_zeroth_moment_trace() -> None:
    """The per-pole weights are the residue traces, so they sum to `Tr[T^0]`."""
    lehmann = build()

    assert float(np.sum(lehmann.weights_per_pole())) == pytest.approx(
        float(np.trace(lehmann.moment(0))), rel=1e-12
    )


@pytest.mark.parametrize("hermitian", [True, False])
def test_non_hermitian_is_handled(hermitian: bool) -> None:
    """Both coupling layouts are accepted."""
    rng = np.random.default_rng(11)
    energies = rng.normal(size=12)
    couplings = rng.normal(size=(4, 12)) if hermitian else rng.normal(size=(2, 4, 12))
    if hermitian:
        couplings[:, :3] *= 1e-11
    else:
        couplings[:, :, :3] *= 1e-11
    lehmann = Lehmann(energies, couplings, sort=False)

    assert lehmann.weightless_poles().count == 3
