"""Tests for the moments of a Lehmann representation.

The moment contraction is written as matrix products rather than as a single ``einsum``,
because the sum over poles appears in all three operands and ``np.einsum`` cannot express
that as a ``tensordot`` even with ``optimize=True``. These tests keep the ``einsum`` form
as the reference and check the products against it, so that the rewrite cannot drift.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from dyson import util
from dyson.representations.enums import Reduction
from dyson.representations.lehmann import Lehmann

if TYPE_CHECKING:
    from dyson.typing import Array

SUBSCRIPTS = {
    Reduction.NONE: "pk,qk,nk->npq",
    Reduction.DIAG: "pk,pk,nk->np",
    Reduction.TRACE: "pk,pk,nk->n",
}


def reference_moments(
    lehmann: Lehmann, order: int | list[int], reduction: Reduction = Reduction.NONE
) -> Array:
    """Moments via the single-``einsum`` route this implementation replaced."""
    squeeze = isinstance(order, int)
    orders = np.asarray([order] if squeeze else order)
    left, right = lehmann.unpack_couplings()
    moments = util.einsum(
        SUBSCRIPTS[Reduction(reduction)],
        right,
        left.conj(),
        lehmann.energies[None] ** orders[:, None],
    )
    return moments[0] if squeeze else moments


def build(
    nphys: int = 6,
    naux: int = 40,
    hermitian: bool = True,
    complex_: bool = False,
    seed: int = 1234,
) -> Lehmann:
    """Build a random Lehmann representation."""
    rng = np.random.default_rng(seed)

    def draw(*shape: int) -> Array:
        out = rng.normal(size=shape)
        if complex_:
            out = out + 1j * rng.normal(size=shape)
        return out

    energies = draw(naux)
    couplings = draw(nphys, naux) if hermitian else draw(2, nphys, naux)
    return Lehmann(energies, couplings)


@pytest.mark.parametrize("reduction", list(Reduction))
@pytest.mark.parametrize("hermitian", [True, False])
@pytest.mark.parametrize("complex_", [True, False])
def test_matches_einsum_reference(
    reduction: Reduction, hermitian: bool, complex_: bool
) -> None:
    """Every reduction agrees with the einsum route it replaced."""
    lehmann = build(hermitian=hermitian, complex_=complex_)
    orders = list(range(6))

    result = lehmann.moments(orders, reduction=reduction)
    expected = reference_moments(lehmann, orders, reduction=reduction)

    assert result.shape == expected.shape
    assert result.dtype == expected.dtype
    np.testing.assert_allclose(result, expected, rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize("reduction", list(Reduction))
def test_scalar_order_is_squeezed(reduction: Reduction) -> None:
    """A scalar order returns a single moment, not a length-one stack."""
    lehmann = build()

    result = lehmann.moments(3, reduction=reduction)
    expected = reference_moments(lehmann, 3, reduction=reduction)

    assert result.shape == expected.shape
    assert result.ndim == Reduction(reduction).ndim
    np.testing.assert_allclose(result, expected, rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize("reduction", list(Reduction))
@pytest.mark.parametrize(
    "orders",
    [
        [0],
        [5, 2, 0, 3],  # unsorted
        [2, 2, 2],  # repeated
        np.arange(8)[::-1],  # a non-contiguous array
        [],  # no orders at all
    ],
    ids=["single", "unsorted", "repeated", "reversed-view", "empty"],
)
def test_order_handling(reduction: Reduction, orders: list[int] | Array) -> None:
    """Orders are honoured in the order given, however the sequence is supplied."""
    lehmann = build()

    result = lehmann.moments(orders, reduction=reduction)
    expected = reference_moments(lehmann, orders, reduction=reduction)

    assert result.shape == expected.shape
    np.testing.assert_allclose(result, expected, rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize("reduction", list(Reduction))
def test_no_poles(reduction: Reduction) -> None:
    """An empty auxiliary space gives zero moments of the right shape."""
    lehmann = build(naux=0)

    result = lehmann.moments(range(4), reduction=reduction)

    assert result.shape == reference_moments(lehmann, list(range(4)), reduction=reduction).shape
    np.testing.assert_allclose(result, 0.0, atol=0.0)


def test_zeroth_moment_is_the_coupling_overlap() -> None:
    """The zeroth moment is the coupling overlap, independent of the pole energies.

    An analytic check that does not go through the reference contraction, so that a shared
    error in both routes would still be caught.
    """
    lehmann = build()
    _, right = lehmann.unpack_couplings()

    np.testing.assert_allclose(
        lehmann.moments(0), right @ right.conj().T, rtol=1e-13, atol=1e-13
    )


def test_first_moment_is_the_energy_weighted_overlap() -> None:
    """The first moment weights each pole's outer product by its energy."""
    lehmann = build()
    _, right = lehmann.unpack_couplings()

    np.testing.assert_allclose(
        lehmann.moments(1),
        (right * lehmann.energies) @ right.conj().T,
        rtol=1e-13,
        atol=1e-13,
    )


@pytest.mark.parametrize("hermitian", [True, False])
def test_reductions_are_consistent_with_each_other(hermitian: bool) -> None:
    """``diag`` is the diagonal of ``none``, and ``trace`` is its trace."""
    lehmann = build(hermitian=hermitian)
    orders = list(range(5))

    full = lehmann.moments(orders, reduction=Reduction.NONE)
    diag = lehmann.moments(orders, reduction=Reduction.DIAG)
    trace = lehmann.moments(orders, reduction=Reduction.TRACE)

    np.testing.assert_allclose(diag, np.diagonal(full, axis1=1, axis2=2), rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(trace, np.trace(full, axis1=1, axis2=2), rtol=1e-13, atol=1e-13)


def test_invalid_reduction_raises() -> None:
    """An unrecognised reduction is rejected rather than silently ignored."""
    lehmann = build()

    with pytest.raises(ValueError):
        lehmann.moments(0, reduction="nonsense")  # type: ignore[arg-type]
