"""Tests for :class:`~dyson.expressions`."""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

import numpy as np
import pyscf
import pytest

from dyson import util
from dyson.expressions import ADC2, CCSD, FCI, HF, ADC2x
from dyson.representations.spectral import Spectral
from dyson.solvers import Davidson, Exact

if TYPE_CHECKING:
    from typing import Any

    from pyscf import scf

    from dyson.expressions.expression import BaseExpression
    from dyson.solvers.solver import StaticSolver

    from .conftest import ExactGetter, Helper


def test_init(mf: scf.hf.RHF, expression_cls: type[BaseExpression]) -> None:
    """Test the instantiation of the expression from a mean-field object."""
    expression = expression_cls.from_mf(mf)
    assert expression.mol is mf.mol
    assert expression.nphys == mf.mol.nao
    assert expression.nocc == mf.mol.nelectron // 2
    assert expression.nvir == mf.mol.nao - mf.mol.nelectron // 2


def test_hamiltonian(mf: scf.hf.RHF, expression_cls: type[BaseExpression]) -> None:
    """Test the Hamiltonian of the expression."""
    expression = expression_cls.from_mf(mf)
    if expression.nconfig > 1024:
        pytest.skip("Skipping test for large Hamiltonian")
    diagonal = expression.diagonal()
    hamiltonian = expression.build_matrix()

    if expression_cls in ADC2._classes:
        # ADC(2)-x diagonal is set to ADC(2) diagonal in PySCF for better Davidson convergence
        assert np.allclose(np.diag(hamiltonian), diagonal)
    assert hamiltonian.shape == expression.shape
    assert (expression.nconfig + expression.nsingle) == diagonal.size

    vector = np.random.random(expression.nconfig + expression.nsingle)
    hv = expression.apply_hamiltonian_right(vector)
    try:
        vh = expression.apply_hamiltonian_left(vector)
    except NotImplementedError:
        vh = None

    assert np.allclose(hv, hamiltonian @ vector)
    if vh is not None:
        assert np.allclose(vh, vector @ hamiltonian)


def test_gf_moments(mf: scf.hf.RHF, expression_cls: type[BaseExpression]) -> None:
    """Test the Green's function moments of the expression."""
    # Get the quantities required from the expression
    expression = expression_cls.from_mf(mf)
    if expression.nconfig > 1024:
        pytest.skip("Skipping test for large Hamiltonian")
    hamiltonian = expression.build_matrix()

    # Construct the moments
    moments = np.zeros((2, expression.nphys, expression.nphys))
    for i, j in itertools.product(range(expression.nphys), repeat=2):
        bra = expression.get_excitation_bra(j)
        ket = expression.get_excitation_ket(i)
        moments[0, j, i] += bra.conj() @ ket
        moments[1, j, i] += bra.conj() @ hamiltonian @ ket

    # Compare the moments to the reference
    ref = expression.build_gf_moments(2)

    assert np.allclose(ref[0], moments[0])
    assert np.allclose(ref[1], moments[1])


def test_static(
    helper: Helper,
    mf: scf.hf.RHF,
    expression_cls: type[BaseExpression],
    exact_cache: ExactGetter,
) -> None:
    """Test the static self-energy of the expression."""
    # Get the quantities required from the expression
    expression = expression_cls.from_mf(mf)
    if expression.nconfig > 1024:
        pytest.skip("Skipping test for large Hamiltonian")
    gf_moments = expression.build_gf_moments(2)

    # Get the static self-energy
    exact = exact_cache(mf, expression_cls)
    assert exact.result is not None
    greens_function = exact.result.get_greens_function()
    static = exact.result.get_static_self_energy()

    assert helper.have_equal_moments(gf_moments, greens_function, 2)
    assert np.allclose(static, gf_moments[1])


@pytest.mark.parametrize(
    "solver_cls, solver_kwargs",
    [
        (Exact, dict()),
        (Davidson, dict(nroots=3)),
    ],
)
def test_hf(
    helper: Helper, mf: scf.hf.RHF, solver_cls: type[StaticSolver], solver_kwargs: dict[str, Any]
) -> None:
    """Test the HF expression."""
    hf_h = HF.h.from_mf(mf)
    hf_p = HF.p.from_mf(mf)
    gf_moments_h = hf_h.build_gf_moments(2)
    gf_moments_p = hf_p.build_gf_moments(2)

    # Get the energy from the hole moments
    h1e = np.einsum("pq,pi,qj->ij", mf.get_hcore(), mf.mo_coeff, mf.mo_coeff)
    energy = util.gf_moments_galitskii_migdal(gf_moments_h, h1e, factor=1.0)

    assert np.abs(energy - mf.energy_elec()[0]) < 1e-8

    # Get the Fock matrix Fock matrix from the moments
    fock_ref = np.einsum("pq,pi,qj->ij", mf.get_fock(), mf.mo_coeff, mf.mo_coeff)
    fock = gf_moments_h[1] + gf_moments_p[1]

    assert np.allclose(fock, fock_ref)
    assert np.allclose(gf_moments_h[1] + gf_moments_p[1], fock)

    # Get the hole Green's function from the Exact solver
    solver_h = solver_cls.from_expression(hf_h, **solver_kwargs)
    solver_h.kernel()
    nocc = mf.mol.nelectron // 2

    assert solver_h.result is not None
    assert np.allclose(
        solver_h.result.get_greens_function().as_perturbed_mo_energy()[nocc - 1],
        mf.mo_energy[nocc - 1],
    )
    if solver_cls is Exact:
        assert helper.have_equal_moments(solver_h.result.get_greens_function(), gf_moments_h, 2)

    # Get the particle Green's function
    solver_p = solver_cls.from_expression(hf_p, **solver_kwargs)
    solver_p.kernel()

    assert solver_p.result is not None
    assert np.allclose(
        solver_p.result.get_greens_function().as_perturbed_mo_energy()[nocc], mf.mo_energy[nocc]
    )
    if solver_cls is Exact:
        assert helper.have_equal_moments(solver_p.result.get_greens_function(), gf_moments_p, 2)

    # Get the combined Green's function
    overlap = hf_h.build_overlap() + hf_p.build_overlap()
    result = Spectral.combine_for_greens_function(solver_h.result, solver_p.result, overlap=overlap)

    assert np.allclose(
        result.get_greens_function().as_perturbed_mo_energy()[nocc - 1], mf.mo_energy[nocc - 1]
    )
    assert np.allclose(
        result.get_greens_function().as_perturbed_mo_energy()[nocc], mf.mo_energy[nocc]
    )
    if solver_cls is Exact:
        assert np.allclose(result.get_greens_function().as_perturbed_mo_energy(), mf.mo_energy)
        assert helper.have_equal_moments(
            result.get_greens_function(), gf_moments_h + gf_moments_p, 2
        )

    # Check the zeroth moments add to identity
    gf_moment_0 = gf_moments_h[0] + gf_moments_p[0]

    assert np.allclose(gf_moment_0, np.eye(mf.mol.nao))


@pytest.mark.parametrize(
    "solver_cls, solver_kwargs",
    [
        (Exact, dict()),
        (Davidson, dict(nroots=3)),
    ],
)
def test_ccsd(
    helper: Helper, mf: scf.hf.RHF, solver_cls: type[StaticSolver], solver_kwargs: dict[str, Any]
) -> None:
    """Test the CCSD expression."""
    ccsd_h = CCSD.h.from_mf(mf)
    ccsd_p = CCSD.p.from_mf(mf)
    pyscf_ccsd = pyscf.cc.CCSD(mf)
    pyscf_ccsd.run(conv_tol=1e-10, conv_tol_normt=1e-8)
    gf_moments_h = ccsd_h.build_gf_moments(2)
    gf_moments_p = ccsd_p.build_gf_moments(2)

    # Get the energy from the hole moments
    h1e = np.einsum("pq,pi,qj->ij", mf.get_hcore(), mf.mo_coeff, mf.mo_coeff)
    energy = util.gf_moments_galitskii_migdal(gf_moments_h, h1e, factor=1.0)
    energy_ref = pyscf_ccsd.e_tot - mf.mol.energy_nuc()

    correct_energy = np.abs(energy - energy_ref) < 1e-8
    if mf.mol.nelectron > 2:
        # Galitskii--Migdal should not capture the energy for CCSD for >2 electrons
        with pytest.raises(AssertionError):
            assert correct_energy
    else:
        assert correct_energy

    # Get the hole Green's function from
    solver_h = solver_cls.from_expression(ccsd_h, **solver_kwargs)
    solver_h.kernel()
    ip_ref, _ = pyscf_ccsd.ipccsd(nroots=3)

    assert solver_h.result is not None
    assert np.allclose(solver_h.result.eigvals[-1], -ip_ref[0])
    if solver_cls is Exact:
        assert helper.have_equal_moments(solver_h.result.get_greens_function(), gf_moments_h, 2)

    # Get the particle Green's function
    solver_p = solver_cls.from_expression(ccsd_p, **solver_kwargs)
    solver_p.kernel()
    ea_ref, _ = pyscf_ccsd.eaccsd(nroots=3)

    assert solver_p.result is not None
    assert np.allclose(solver_p.result.eigvals[0], ea_ref[0])
    if solver_cls is Exact:
        assert helper.have_equal_moments(solver_p.result.get_greens_function(), gf_moments_p, 2)

    # Get the combined Green's function
    overlap = ccsd_h.build_overlap() + ccsd_p.build_overlap()
    result = Spectral.combine_for_greens_function(solver_h.result, solver_p.result, overlap=overlap)

    assert np.allclose(result.get_greens_function().occupied().energies[-1], -ip_ref[0])
    assert np.allclose(result.get_greens_function().virtual().energies[0], ea_ref[0])
    if solver_cls is Exact:
        assert helper.have_equal_moments(
            result.get_greens_function(), gf_moments_h + gf_moments_p, 2
        )

    # Check the RDM
    rdm1 = gf_moments_h[0].copy()
    rdm1 += rdm1.T.conj()
    rdm1_ref = pyscf_ccsd.make_rdm1(with_mf=True)

    assert np.allclose(rdm1, rdm1_ref)

    # Check the zeroth moments add to identity
    gf_moment_0 = gf_moments_h[0] + gf_moments_p[0]

    assert np.allclose(gf_moment_0, np.eye(mf.mol.nao))


@pytest.mark.parametrize(
    "solver_cls, solver_kwargs",
    [
        (Exact, dict()),
        (Davidson, dict(nroots=3)),
    ],
)
def test_fci(
    helper: Helper, mf: scf.hf.RHF, solver_cls: type[StaticSolver], solver_kwargs: dict[str, Any]
) -> None:
    """Test the FCI expression."""
    fci_h = FCI.h.from_mf(mf)
    fci_p = FCI.p.from_mf(mf)
    pyscf_fci = pyscf.fci.FCI(mf)
    gf_moments_h = fci_h.build_gf_moments(2)
    gf_moments_p = fci_p.build_gf_moments(2)

    # Get the energy from the hole moments
    h1e = np.einsum("pq,pi,qj->ij", mf.get_hcore(), mf.mo_coeff, mf.mo_coeff)
    energy = util.gf_moments_galitskii_migdal(gf_moments_h, h1e, factor=1.0)
    energy_fci = pyscf_fci.kernel()[0]
    energy_ref = energy_fci - mf.mol.energy_nuc()

    assert np.abs(energy - energy_ref) < 1e-8

    # Get the hole Green's function
    solver_h = solver_cls.from_expression(fci_h, **solver_kwargs)
    solver_h.kernel()
    mol_h = mf.mol.copy()
    mol_h.nelec = (mf.mol.nelec[0] - 1, mf.mol.nelec[1])
    pyscf_fci_h = pyscf.fci.FCI(mf.__class__(mol_h).run())
    energy_fci_h = pyscf_fci_h.kernel()[0]
    energy_ref_h = energy_fci_h - mf.mol.energy_nuc()
    ip_ref = energy_ref - energy_ref_h

    assert solver_h.result is not None
    assert np.allclose(solver_h.result.eigvals[-1], ip_ref)
    if solver_cls is Exact:
        assert helper.have_equal_moments(solver_h.result.get_greens_function(), gf_moments_h, 2)

    # Get the particle Green's function
    solver_p = solver_cls.from_expression(fci_p, **solver_kwargs)
    solver_p.kernel()
    mol_p = mf.mol.copy()
    mol_p.nelec = (mf.mol.nelec[0] + 1, mf.mol.nelec[1])
    pyscf_fci_p = pyscf.fci.FCI(mf.__class__(mol_p).run())
    energy_fci_p = pyscf_fci_p.kernel()[0]
    energy_ref_p = energy_fci_p - mf.mol.energy_nuc()
    ea_ref = energy_ref_p - energy_ref

    assert solver_p.result is not None
    assert np.allclose(solver_p.result.eigvals[0], ea_ref)
    if solver_cls is Exact:
        assert helper.have_equal_moments(solver_p.result.get_greens_function(), gf_moments_p, 2)

    # Get the combined Green's function
    overlap = fci_h.build_overlap() + fci_p.build_overlap()
    result = Spectral.combine_for_greens_function(solver_h.result, solver_p.result, overlap=overlap)

    assert np.allclose(result.get_greens_function().occupied().energies[-1], ip_ref)
    assert np.allclose(result.get_greens_function().virtual().energies[0], ea_ref)
    if solver_cls is Exact:
        assert helper.have_equal_moments(
            result.get_greens_function(), gf_moments_h + gf_moments_p, 2
        )

    # Check the RDM
    rdm1 = fci_h.build_gf_moments(1)[0] * 2
    rdm1_ref = pyscf_fci.make_rdm1(pyscf_fci.ci, mf.mol.nao, mf.mol.nelectron)

    assert np.allclose(rdm1, rdm1_ref)

    # Check the zeroth moments add to identity
    gf_moment_0 = gf_moments_h[0] + gf_moments_p[0]

    assert np.allclose(gf_moment_0, np.eye(mf.mol.nao))

    if mf.mol.nelectron <= 2:
        # CCSD should match FCI for <=2 electrons for the hole moments
        ccsd = CCSD.h.from_mf(mf)
        gf_moments_ccsd = ccsd.build_gf_moments(2)

        assert np.allclose(gf_moments_ccsd[0], gf_moments_h[0])
        assert np.allclose(gf_moments_ccsd[1], gf_moments_h[1])


@pytest.mark.parametrize(
    "solver_cls, solver_kwargs",
    [
        (Exact, dict()),
        (Davidson, dict(nroots=3)),
    ],
)
def test_adc2(
    helper: Helper, mf: scf.hf.RHF, solver_cls: type[StaticSolver], solver_kwargs: dict[str, Any]
) -> None:
    """Test the ADC(2) expression."""
    adc_h = ADC2.h.from_mf(mf)
    adc_p = ADC2.p.from_mf(mf)
    pyscf_adc_ip = pyscf.adc.ADC(mf)
    pyscf_adc_ea = pyscf.adc.ADC(mf)
    pyscf_adc_ea.method_type = "ea"
    gf_moments_h = adc_h.build_gf_moments(2)
    gf_moments_p = adc_p.build_gf_moments(2)

    # Get the energy from the hole moments
    h1e = np.einsum("pq,pi,qj->ij", mf.get_hcore(), mf.mo_coeff, mf.mo_coeff)
    energy = util.gf_moments_galitskii_migdal(gf_moments_h, h1e, factor=1.0)
    energy_ref = mf.energy_elec()[0] + pyscf_adc_ip.kernel_gs()[0]

    assert np.abs(energy - energy_ref) < 1e-8

    # Get the hole Green's function from the Davidson solver
    solver_h = solver_cls.from_expression(adc_h, **solver_kwargs)
    solver_h.kernel()
    ip_ref, _, _, _ = pyscf_adc_ip.kernel(nroots=3)

    assert solver_h.result is not None
    assert np.allclose(solver_h.result.eigvals[-1], -ip_ref[0])
    if solver_cls is Exact:
        assert helper.have_equal_moments(solver_h.result.get_greens_function(), gf_moments_h, 2)

    # Get the particle Green's function from the Davidson solver
    solver_p = solver_cls.from_expression(adc_p, **solver_kwargs)
    solver_p.kernel()
    ea_ref, _, _, _ = pyscf_adc_ea.kernel(nroots=3)

    assert solver_p.result is not None
    assert np.allclose(solver_p.result.eigvals[0], ea_ref[0])
    if solver_cls is Exact:
        assert helper.have_equal_moments(solver_p.result.get_greens_function(), gf_moments_p, 2)

    # Get the combined Green's function from the Davidson solver
    overlap = adc_h.build_overlap() + adc_p.build_overlap()
    result = Spectral.combine_for_greens_function(solver_h.result, solver_p.result, overlap=overlap)

    assert np.allclose(result.get_greens_function().occupied().energies[-1], -ip_ref[0])
    assert np.allclose(result.get_greens_function().virtual().energies[0], ea_ref[0])
    if solver_cls is Exact:
        assert helper.have_equal_moments(
            result.get_greens_function(), gf_moments_h + gf_moments_p, 2
        )

    # Check the RDM
    rdm1 = adc_h.build_gf_moments(1)[0] * 2
    rdm1_ref = np.diag(mf.mo_occ)  # No correlated ground state!

    assert np.allclose(rdm1, rdm1_ref)

    # Check the zeroth moments add to identity
    gf_moment_0 = gf_moments_h[0] + gf_moments_p[0]

    assert np.allclose(gf_moment_0, np.eye(mf.mol.nao))


@pytest.mark.parametrize(
    "solver_cls, solver_kwargs",
    [
        (Exact, dict()),
        (Davidson, dict(nroots=3)),
    ],
)
def test_adc2x(
    helper: Helper, mf: scf.hf.RHF, solver_cls: type[StaticSolver], solver_kwargs: dict[str, Any]
) -> None:
    """Test the ADC(2)-x expression."""
    adc_h = ADC2x.h.from_mf(mf)
    adc_p = ADC2x.p.from_mf(mf)
    pyscf_adc_ip = pyscf.adc.ADC(mf)
    pyscf_adc_ip.method = "adc(2)-x"
    pyscf_adc_ea = pyscf.adc.ADC(mf)
    pyscf_adc_ea.method = "adc(2)-x"
    pyscf_adc_ea.method_type = "ea"
    gf_moments_h = adc_h.build_gf_moments(2)
    gf_moments_p = adc_p.build_gf_moments(2)

    # Get the energy from the hole moments
    h1e = np.einsum("pq,pi,qj->ij", mf.get_hcore(), mf.mo_coeff, mf.mo_coeff)
    energy = util.gf_moments_galitskii_migdal(gf_moments_h, h1e, factor=1.0)
    energy_ref = mf.energy_elec()[0] + pyscf_adc_ip.kernel_gs()[0]

    assert np.abs(energy - energy_ref) < 1e-8

    # Get the hole Green's function from the Davidson solver
    solver_h = solver_cls.from_expression(adc_h, **solver_kwargs)
    solver_h.kernel()
    ip_ref, _, _, _ = pyscf_adc_ip.kernel(nroots=3)

    assert solver_h.result is not None
    assert np.allclose(solver_h.result.eigvals[-1], -ip_ref[0])
    if solver_cls is Exact:
        assert helper.have_equal_moments(solver_h.result.get_greens_function(), gf_moments_h, 2)

    # Get the particle Green's function from the Davidson solver
    solver_p = solver_cls.from_expression(adc_p, **solver_kwargs)
    solver_p.kernel()
    ea_ref, _, _, _ = pyscf_adc_ea.kernel(nroots=3)

    assert solver_p.result is not None
    assert np.allclose(solver_p.result.eigvals[0], ea_ref[0])
    if solver_cls is Exact:
        assert helper.have_equal_moments(solver_p.result.get_greens_function(), gf_moments_p, 2)

    # Get the combined Green's function from the Davidson solver
    overlap = adc_h.build_overlap() + adc_p.build_overlap()
    result = Spectral.combine_for_greens_function(solver_h.result, solver_p.result, overlap=overlap)

    assert np.allclose(result.get_greens_function().occupied().energies[-1], -ip_ref[0])
    assert np.allclose(result.get_greens_function().virtual().energies[0], ea_ref[0])
    if solver_cls is Exact:
        assert helper.have_equal_moments(
            result.get_greens_function(), gf_moments_h + gf_moments_p, 2
        )

    # Check the RDM
    rdm1 = adc_h.build_gf_moments(1)[0] * 2
    rdm1_ref = np.diag(mf.mo_occ)  # No correlated ground state!

    assert np.allclose(rdm1, rdm1_ref)

    # Check the zeroth moments add to identity
    gf_moment_0 = gf_moments_h[0] + gf_moments_p[0]

    assert np.allclose(gf_moment_0, np.eye(mf.mol.nao))
