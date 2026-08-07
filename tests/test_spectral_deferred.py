"""Tests for the deferred eigendecomposition of :cls:`Spectral`.

A spectrum built by :meth:`Spectral.from_self_energy` can serve the static part, the
self-energy and the overlap without diagonalising the supermatrix, because reconstructing
a block of that matrix from its own eigendecomposition returns the block it was built
from. These tests pin both halves of that: that the values are right, and that getting
them did not diagonalise anything.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from dyson.representations.lehmann import Lehmann
from dyson.representations.spectral import Spectral

if TYPE_CHECKING:
    from dyson.typing import Array


def build(
    nphys: int = 5,
    naux: int = 30,
    *,
    hermitian: bool = True,
    complex_: bool = False,
    degenerate: bool = False,
    seed: int = 90210,
) -> tuple[Array, Lehmann]:
    """Build a random static part and auxiliary Lehmann representation."""
    rng = np.random.default_rng(seed)

    def draw(*shape: int) -> Array:
        out = rng.normal(size=shape)
        if complex_:
            out = out + 1j * rng.normal(size=shape)
        return out

    static = draw(nphys, nphys)
    static = static + static.T.conj()
    energies = rng.normal(size=naux)
    if degenerate:
        # Repeat a handful of energies exactly, so the eigenvectors of the auxiliary block
        # are only defined up to a rotation within each degenerate group.
        half = 2 * (naux // 4)
        energies[:half] = np.repeat(energies[: half // 2], 2)
    couplings = draw(nphys, naux) if hermitian else draw(2, nphys, naux)
    return static, Lehmann(energies, couplings)


def overlap_matrix(nphys: int, seed: int = 5) -> Array:
    """A positive-definite, non-identity physical-space overlap."""
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(nphys, nphys))
    return a @ a.T + nphys * np.eye(nphys)


PARAMS = [
    pytest.param(True, False, False, id="hermitian"),
    pytest.param(True, True, False, id="hermitian-complex"),
    pytest.param(False, False, False, id="non-hermitian"),
    pytest.param(False, True, False, id="non-hermitian-complex"),
    pytest.param(True, False, True, id="hermitian-degenerate"),
    pytest.param(False, False, True, id="non-hermitian-degenerate"),
]


@pytest.mark.parametrize(("hermitian", "complex_", "degenerate"), PARAMS)
@pytest.mark.parametrize("with_overlap", [False, True], ids=["no-overlap", "overlap"])
def test_inputs_returned_without_diagonalising(
    hermitian: bool, complex_: bool, degenerate: bool, with_overlap: bool
) -> None:
    """The static part, self-energy and overlap come back, and nothing is diagonalised."""
    static, self_energy = build(hermitian=hermitian, complex_=complex_, degenerate=degenerate)
    overlap = overlap_matrix(static.shape[0]) if with_overlap else None
    spectral = Spectral.from_self_energy(static, self_energy, overlap=overlap)

    assert not spectral.diagonalised

    np.testing.assert_allclose(spectral.get_static_self_energy(), static, rtol=0, atol=0)

    recovered = spectral.get_self_energy()
    np.testing.assert_allclose(recovered.energies, self_energy.energies, rtol=0, atol=0)
    np.testing.assert_allclose(recovered.couplings, self_energy.couplings, rtol=0, atol=0)

    expected_overlap = overlap if with_overlap else np.eye(static.shape[0])
    np.testing.assert_allclose(spectral.overlap, expected_overlap, rtol=0, atol=0)

    assert spectral.hermitian == hermitian
    assert spectral.nphys == static.shape[0]
    assert spectral.neig == static.shape[0] + self_energy.naux

    # The whole point: none of the above ran an eigendecomposition.
    assert not spectral.diagonalised


@pytest.mark.parametrize(("hermitian", "complex_", "degenerate"), PARAMS)
@pytest.mark.parametrize("with_overlap", [False, True], ids=["no-overlap", "overlap"])
def test_spectrum_matches_without_eigenvectors(
    hermitian: bool, complex_: bool, degenerate: bool, with_overlap: bool
) -> None:
    """The eigenvalues-only route agrees with the full one and computes no eigenvectors."""
    static, self_energy = build(hermitian=hermitian, complex_=complex_, degenerate=degenerate)
    overlap = overlap_matrix(static.shape[0]) if with_overlap else None
    spectral = Spectral.from_self_energy(static, self_energy, overlap=overlap)

    spectrum = spectral.spectrum
    assert not spectral.diagonalised
    assert spectrum.shape[0] == spectral.neig

    reference = Spectral.from_self_energy(static, self_energy, overlap=overlap).eigvals
    np.testing.assert_allclose(np.sort(spectrum), np.sort(reference), rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize(("hermitian", "complex_", "degenerate"), PARAMS)
@pytest.mark.parametrize("with_overlap", [False, True], ids=["no-overlap", "overlap"])
def test_deferred_agrees_with_eager(
    hermitian: bool, complex_: bool, degenerate: bool, with_overlap: bool
) -> None:
    """Reconstructing through the eigendecomposition gives the same answers.

    This is the property the deferral relies on, checked against the route it replaced
    rather than assumed.
    """
    static, self_energy = build(hermitian=hermitian, complex_=complex_, degenerate=degenerate)
    overlap = overlap_matrix(static.shape[0]) if with_overlap else None

    deferred = Spectral.from_self_energy(static, self_energy, overlap=overlap)
    eager = Spectral(
        *self_energy.diagonalise_matrix(static, overlap=overlap),
        self_energy.nphys,
        chempot=self_energy.chempot,
    )

    np.testing.assert_allclose(
        deferred.get_static_self_energy(), eager.get_static_self_energy(), rtol=1e-9, atol=1e-9
    )
    a, b = deferred.get_self_energy(), eager.get_self_energy()
    np.testing.assert_allclose(np.sort(a.energies), np.sort(b.energies), rtol=1e-9, atol=1e-9)
    # Couplings are defined only up to a rotation within degenerate groups, so compare the
    # moments, which are invariant under one.
    np.testing.assert_allclose(a.moments(range(4)), b.moments(range(4)), rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(deferred.overlap, eager.overlap, rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize(
    "accessor",
    [
        pytest.param(lambda s: s.eigvals, id="eigvals"),
        pytest.param(lambda s: s.eigvecs, id="eigvecs"),
        pytest.param(lambda s: s.get_dyson_orbitals(), id="get_dyson_orbitals"),
        pytest.param(lambda s: s.get_greens_function(), id="get_greens_function"),
        pytest.param(lambda s: s.get_auxiliaries(), id="get_auxiliaries"),
    ],
)
def test_eigenpair_accessors_do_diagonalise(accessor) -> None:  # type: ignore[no-untyped-def]
    """Anything that genuinely needs an eigenpair triggers the decomposition."""
    static, self_energy = build()
    spectral = Spectral.from_self_energy(static, self_energy)

    assert not spectral.diagonalised
    accessor(spectral)
    assert spectral.diagonalised


def test_realisation_is_idempotent_and_consistent() -> None:
    """Forcing the decomposition does not change what the cheap accessors report."""
    static, self_energy = build()
    overlap = overlap_matrix(static.shape[0])
    spectral = Spectral.from_self_energy(static, self_energy, overlap=overlap)

    before_static = spectral.get_static_self_energy().copy()
    before_overlap = np.array(spectral.overlap)
    _ = spectral.eigvals
    assert spectral.diagonalised

    np.testing.assert_allclose(
        spectral.get_static_self_energy(), before_static, rtol=1e-9, atol=1e-9
    )
    np.testing.assert_allclose(spectral.overlap, before_overlap, rtol=1e-9, atol=1e-9)
    _ = spectral.eigvecs  # a second access must not redo or disturb anything
    assert spectral.diagonalised


def test_sort_is_honoured_on_realisation() -> None:
    """A deferred spectrum built with ``sort=True`` is sorted once it is realised."""
    static, self_energy = build()
    spectral = Spectral.from_self_energy(static, self_energy, sort=True)

    assert not spectral.diagonalised
    assert np.all(np.diff(spectral.eigvals) >= 0)


def test_chempot_is_carried_and_overridable() -> None:
    """``get_self_energy`` honours both the stored and an explicit chemical potential."""
    static, self_energy = build()
    self_energy = self_energy.copy(chempot=0.25)
    spectral = Spectral.from_self_energy(static, self_energy)

    assert spectral.get_self_energy().chempot == 0.25
    assert spectral.get_self_energy(chempot=-1.5).chempot == -1.5
    assert not spectral.diagonalised


def test_returned_self_energy_is_a_copy() -> None:
    """Mutating what comes out must not corrupt the spectrum's own source."""
    static, self_energy = build()
    spectral = Spectral.from_self_energy(static, self_energy)

    recovered = spectral.get_self_energy()
    recovered.energies[0] += 1.0

    np.testing.assert_allclose(
        spectral.get_self_energy().energies, self_energy.energies, rtol=0, atol=0
    )


def test_combine_for_self_energy_concatenates() -> None:
    """Combining two sectors gives the union of their auxiliaries."""
    static, occupied = build(naux=20, seed=1)
    _, virtual = build(naux=14, seed=2)
    a = Spectral.from_self_energy(static, occupied)
    b = Spectral.from_self_energy(static, virtual)

    combined = Spectral.combine_for_self_energy(a, b).get_self_energy()

    assert combined.naux == occupied.naux + virtual.naux
    np.testing.assert_allclose(
        np.sort(combined.energies),
        np.sort(np.concatenate([occupied.energies, virtual.energies])),
        rtol=1e-9,
        atol=1e-9,
    )
