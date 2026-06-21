"""Unrestricted coupled cluster singles and doubles (UCCSD) expressions.

Spin-blocked (UHF reference) analogue of the CCSD expressions in :mod:`dyson.expressions.ccsd`.

PySCF's :mod:`pyscf.cc.eom_uccsd` only provides the *right* EOM matrix-vector products
(``ipccsd_matvec``, ``eaccsd_matvec``) and not their transposes (``lipccsd_matvec``,
``leaccsd_matvec``). A moment :math:`\\langle x | H^n | y \\rangle` is a bilinear form, so it can be
evaluated by propagating either side -- :math:`\\langle x | H^n | y \\rangle =
((H^\\mathsf{T})^n x)^\\mathsf{T} y` -- and the available right matvec is the transpose of the
missing left one. Each sector is therefore wired so that the *available* matvec backs the direction
the moment machinery calls: the hole sector applies the Hamiltonian to the bra
(:func:`~dyson.expressions.expression.BaseExpression.build_gf_moments` with ``left=True``), and the
particle sector to the ket (``left=False``, the default).

The excitation vectors are evaluated directly in the spin-blocked (alpha/beta) UHF representation.
The formulae are the spin-resolution of the spin-orbital expressions in
:mod:`dyson.expressions.gccsd`, and are validated against that (spin-orbital) construction.
"""

from __future__ import annotations

import warnings
from abc import abstractmethod
from typing import TYPE_CHECKING

from pyscf import cc, scf

from dyson import numpy as np
from dyson import util
from dyson.expressions.expression import BaseUExpression, ExpressionCollection
from dyson.representations.enums import Reduction

if TYPE_CHECKING:
    from typing import Any

    from pyscf.gto.mole import Mole
    from pyscf.scf.hf import SCF

    from dyson.typing import Array


# --- Native spin-blocked excitation-vector amplitudes -------------------------------------------
#
# The physical orbital index runs over the alpha block (``nmoa`` orbitals) followed by the beta
# block (``nmob`` orbitals); within each block the occupied orbitals precede the virtuals. Each
# helper returns ``(r1, r2)`` block tuples in the PySCF UCCSD layout:
#   IP:  r1 = (r1a, r1b),  r2 = (r2aaa, r2baa, r2abb, r2bbb)
#   EA:  r1 = (r1a, r1b),  r2 = (r2aaa, r2aba, r2bab, r2bbb)
# These are the spin-resolution of the spin-orbital formulae in :mod:`dyson.expressions.gccsd`.


def _classify(orbital: int, nocc: tuple[int, int], nvir: tuple[int, int]) -> tuple[str, int]:
    """Classify a physical orbital as ``ao``/``av``/``bo``/``bv`` and return its block index."""
    nocca, noccb = nocc
    nvira, _ = nvir
    nmoa = nocca + nvira
    if orbital < nocca:
        return "ao", orbital
    if orbital < nmoa:
        return "av", orbital - nocca
    if orbital < nmoa + noccb:
        return "bo", orbital - nmoa
    return "bv", orbital - nmoa - noccb


def _ip_blocks(nocc: tuple[int, int], nvir: tuple[int, int]) -> tuple[Array, Array, Array, Array, Array, Array]:
    """Allocate zeroed IP amplitude blocks."""
    nocca, noccb = nocc
    nvira, nvirb = nvir
    return (
        np.zeros(nocca),
        np.zeros(noccb),
        np.zeros((nocca, nocca, nvira)),
        np.zeros((noccb, nocca, nvira)),
        np.zeros((nocca, noccb, nvirb)),
        np.zeros((noccb, noccb, nvirb)),
    )


def _ea_blocks(nocc: tuple[int, int], nvir: tuple[int, int]) -> tuple[Array, Array, Array, Array, Array, Array]:
    """Allocate zeroed EA amplitude blocks."""
    nocca, noccb = nocc
    nvira, nvirb = nvir
    return (
        np.zeros(nvira),
        np.zeros(nvirb),
        np.zeros((nocca, nvira, nvira)),
        np.zeros((nocca, nvirb, nvira)),
        np.zeros((noccb, nvira, nvirb)),
        np.zeros((noccb, nvirb, nvirb)),
    )


def _ip_bra(t1: Any, t2: Any, l1: Any, l2: Any, nocc: tuple[int, int], nvir: tuple[int, int], orbital: int) -> tuple[Any, Any]:
    """IP bra (right) amplitudes for a physical orbital."""
    t1a, t1b = t1
    t2aa, t2ab, t2bb = t2
    r1a, r1b, r2aaa, r2baa, r2abb, r2bbb = _ip_blocks(nocc, nvir)
    kind, idx = _classify(orbital, nocc, nvir)

    if kind == "ao":
        r1a[idx] = 1.0
    elif kind == "av":
        r1a[:] = t1a[:, idx]
        r2aaa[:] = t2aa[:, :, idx]
        r2abb[:] = t2ab[:, :, idx, :]
    elif kind == "bo":
        r1b[idx] = 1.0
    else:  # bv
        r1b[:] = t1b[:, idx]
        r2bbb[:] = t2bb[:, :, idx]
        r2baa[:] = util.einsum("iJb->Jib", t2ab[:, :, :, idx])

    return (r1a, r1b), (r2aaa, r2baa, r2abb, r2bbb)


def _ip_ket(t1: Any, t2: Any, l1: Any, l2: Any, nocc: tuple[int, int], nvir: tuple[int, int], orbital: int) -> tuple[Any, Any]:
    """IP ket (left/Lambda) amplitudes for a physical orbital."""
    t1a, t1b = t1
    t2aa, t2ab, t2bb = t2
    l1a, l1b = l1
    l2aa, l2ab, l2bb = l2
    nocca, noccb = nocc
    r1a, r1b, r2aaa, r2baa, r2abb, r2bbb = _ip_blocks(nocc, nvir)
    kind, idx = _classify(orbital, nocc, nvir)

    if kind == "ao":
        i = idx
        r1a[i] = 1.0
        r1a -= util.einsum("ie,e->i", l1a, t1a[i])
        r1a -= 0.5 * util.einsum("imef,mef->i", l2aa, t2aa[i])
        r1a -= util.einsum("iMeF,MeF->i", l2ab, t2ab[i])
        tmp = -0.5 * util.einsum("ijea,e->ija", l2aa, t1a[i])
        r2aaa = tmp - tmp.swapaxes(0, 1)
        tmp = util.einsum("ja,i->ija", l1a, np.eye(nocca)[i])
        r2aaa += tmp - tmp.swapaxes(0, 1)
        r2abb = -util.einsum("iJeB,e->iJB", l2ab, t1a[i])
        r2abb += util.einsum("JB,i->iJB", l1b, np.eye(nocca)[i])
    elif kind == "av":
        r1a[:] = l1a[:, idx]
        r2aaa[:] = l2aa[:, :, idx]
        r2abb[:] = l2ab[:, :, idx, :]
    elif kind == "bo":
        i = idx
        r1b[i] = 1.0
        r1b -= util.einsum("IE,E->I", l1b, t1b[i])
        r1b -= 0.5 * util.einsum("IMEF,MEF->I", l2bb, t2bb[i])
        r1b -= util.einsum("mIeF,meF->I", l2ab, t2ab[:, i])
        tmp = -0.5 * util.einsum("IJEA,E->IJA", l2bb, t1b[i])
        r2bbb = tmp - tmp.swapaxes(0, 1)
        tmp = util.einsum("JA,I->IJA", l1b, np.eye(noccb)[i])
        r2bbb += tmp - tmp.swapaxes(0, 1)
        r2baa = -util.einsum("iJbE,E->Jib", l2ab, t1b[i])
        r2baa += util.einsum("ib,J->Jib", l1a, np.eye(noccb)[i])
    else:  # bv
        r1b[:] = l1b[:, idx]
        r2bbb[:] = l2bb[:, :, idx]
        r2baa[:] = util.einsum("iJb->Jib", l2ab[:, :, :, idx])

    return (r1a, r1b), (r2aaa, r2baa, r2abb, r2bbb)


def _ea_bra(t1: Any, t2: Any, l1: Any, l2: Any, nocc: tuple[int, int], nvir: tuple[int, int], orbital: int) -> tuple[Any, Any]:
    """EA bra (left/Lambda) amplitudes for a physical orbital."""
    t1a, t1b = t1
    t2aa, t2ab, t2bb = t2
    l1a, l1b = l1
    l2aa, l2ab, l2bb = l2
    nvira, nvirb = nvir
    r1a, r1b, r2aaa, r2aba, r2bab, r2bbb = _ea_blocks(nocc, nvir)
    kind, idx = _classify(orbital, nocc, nvir)

    if kind == "ao":
        i = idx
        r1a[:] = -l1a[i]
        r2aaa[:] = -l2aa[i]
        r2bab[:] = -l2ab[i]
    elif kind == "bo":
        i = idx
        r1b[:] = -l1b[i]
        r2bbb[:] = -l2bb[i]
        r2aba[:] = -util.einsum("jaB->jBa", l2ab[:, i])
    elif kind == "av":
        a = idx
        r1a[a] = 1.0
        r1a -= util.einsum("mb,m->b", l1a, t1a[:, a])
        r1a -= 0.5 * util.einsum("kmeb,kme->b", l2aa, t2aa[:, :, :, a])
        r1a -= util.einsum("kMbE,kME->b", l2ab, t2ab[:, :, a, :])
        tmp = -0.5 * util.einsum("ikba,k->iab", l2aa, t1a[:, a])
        r2aaa = tmp - tmp.swapaxes(1, 2)
        tmp = util.einsum("ib,a->iab", l1a, np.eye(nvira)[a])
        r2aaa += tmp - tmp.swapaxes(1, 2)
        r2bab = -util.einsum("kIaB,k->IaB", l2ab, t1a[:, a])
        r2bab += util.einsum("IB,a->IaB", l1b, np.eye(nvira)[a])
    else:  # bv
        a = idx
        r1b[a] = 1.0
        r1b -= util.einsum("MB,M->B", l1b, t1b[:, a])
        r1b -= 0.5 * util.einsum("KMEB,KME->B", l2bb, t2bb[:, :, :, a])
        r1b -= util.einsum("mKeB,mKe->B", l2ab, t2ab[:, :, :, a])
        tmp = -0.5 * util.einsum("IKBA,K->IAB", l2bb, t1b[:, a])
        r2bbb = tmp - tmp.swapaxes(1, 2)
        tmp = util.einsum("IB,A->IAB", l1b, np.eye(nvirb)[a])
        r2bbb += tmp - tmp.swapaxes(1, 2)
        r2aba = -util.einsum("iKaB,K->iBa", l2ab, t1b[:, a])
        r2aba += util.einsum("ia,B->iBa", l1a, np.eye(nvirb)[a])

    return (r1a, r1b), (r2aaa, r2aba, r2bab, r2bbb)


def _ea_ket(t1: Any, t2: Any, l1: Any, l2: Any, nocc: tuple[int, int], nvir: tuple[int, int], orbital: int) -> tuple[Any, Any]:
    """EA ket (right) amplitudes for a physical orbital (negated, as in the spin-orbital case)."""
    t1a, t1b = t1
    t2aa, t2ab, t2bb = t2
    r1a, r1b, r2aaa, r2aba, r2bab, r2bbb = _ea_blocks(nocc, nvir)
    kind, idx = _classify(orbital, nocc, nvir)

    if kind == "ao":
        i = idx
        r1a[:] = t1a[i]
        r2aaa[:] = t2aa[i]
        r2bab[:] = t2ab[i]
    elif kind == "bo":
        i = idx
        r1b[:] = t1b[i]
        r2bbb[:] = t2bb[i]
        r2aba[:] = util.einsum("jaB->jBa", t2ab[:, i])
    elif kind == "av":
        r1a[idx] = -1.0
    else:  # bv
        r1b[idx] = -1.0

    r1 = (-r1a, -r1b)
    r2 = (-r2aaa, -r2aba, -r2bab, -r2bbb)
    return r1, r2


class BaseUCCSD(BaseUExpression):
    """Base class for UCCSD expressions.

    Notes:
        ``nocc`` and ``nmo`` are ``(alpha, beta)`` tuples, as required by the PySCF UCCSD EOM
        routines. The physical Green's function dimension :attr:`nphys` is the total number of
        spin-orbitals (alpha block followed by beta block).
    """

    hermitian_downfolded = False
    hermitian_upfolded = False

    partition: str | None = None

    PYSCF_EOM = cc.eom_uccsd

    def __init__(
        self,
        mol: Mole,
        ccsd: cc.uccsd.UCCSD,
        imds: Any,
    ):
        """Initialise the expression.

        Args:
            mol: Molecule object.
            ccsd: UCCSD object (retained for interoperation with PySCF EOM routines).
            imds: Intermediate integrals.
        """
        self._mol = mol
        self._cc = ccsd
        self._imds = imds
        self.max_memory = ccsd.max_memory
        self._t1 = ccsd.t1
        self._t2 = ccsd.t2
        self._l1 = ccsd.l1
        self._l2 = ccsd.l2
        self._precompute_imds()

    @abstractmethod
    def _precompute_imds(self) -> None:
        """Precompute intermediate integrals."""
        pass

    @classmethod
    def from_ccsd(cls, ccsd: cc.uccsd.UCCSD) -> BaseUCCSD:
        """Create an expression from a UCCSD object.

        Args:
            ccsd: UCCSD object.

        Returns:
            Expression object.
        """
        if not ccsd.converged:
            warnings.warn("UCCSD T amplitudes are not converged.", UserWarning, stacklevel=2)
        if not ccsd.converged_lambda:
            warnings.warn("UCCSD L amplitudes are not converged.", UserWarning, stacklevel=2)
        eris = ccsd.ao2mo()
        imds = cls.PYSCF_EOM._IMDS(ccsd, eris=eris)  # pylint: disable=protected-access
        return cls(
            mol=ccsd._scf.mol,  # pylint: disable=protected-access
            ccsd=ccsd,
            imds=imds,
        )

    @classmethod
    def from_mf(cls, mf: SCF) -> BaseUCCSD:
        """Create an expression from a mean-field object.

        Args:
            mf: Mean-field object. If it is not an unrestricted (UHF) mean-field, it is converted to
                one via ``mf.to_uhf()``.

        Returns:
            Expression object.
        """
        if not isinstance(mf, scf.uhf.UHF):
            mf = mf.to_uhf()
        ccsd = cc.UCCSD(mf)
        ccsd.conv_tol_normt = 1e-9
        ccsd.kernel()
        ccsd.solve_lambda()
        return cls.from_ccsd(ccsd)

    @abstractmethod
    def vector_to_amplitudes(self, vector: Array, *args: Any) -> tuple[Any, Any]:
        """Convert a vector to amplitudes."""
        pass

    @abstractmethod
    def amplitudes_to_vector(self, r1: Any, r2: Any) -> Array:
        """Convert amplitudes to a vector."""
        pass

    def build_se_moments(self, nmom: int, reduction: Reduction = Reduction.NONE) -> Array:
        """Build the self-energy moments."""
        raise NotImplementedError("Self-energy moments not implemented for UCCSD.")

    @property
    def mol(self) -> Mole:
        """Molecule object."""
        return self._mol

    @property
    def t1(self) -> Any:
        """T1 amplitudes (``(alpha, beta)``)."""
        return self._t1

    @property
    def t2(self) -> Any:
        """T2 amplitudes (``(aa, ab, bb)``)."""
        return self._t2

    @property
    def l1(self) -> Any:
        """L1 amplitudes (``(alpha, beta)``)."""
        return self._l1

    @property
    def l2(self) -> Any:
        """L2 amplitudes (``(aa, ab, bb)``)."""
        return self._l2

    @property
    def non_dyson(self) -> bool:
        """Whether the expression produces a non-Dyson Green's function."""
        return False

    # --- orbital counts (tuples for PySCF interoperation; nphys for the physical space) ---

    @property
    def nocc(self) -> tuple[int, int]:
        """Number of occupied spin-orbitals ``(alpha, beta)``."""
        nocca, noccb = self._cc.get_nocc()
        return int(nocca), int(noccb)

    @property
    def nmo(self) -> tuple[int, int]:
        """Number of molecular orbitals ``(alpha, beta)``."""
        nmoa, nmob = self._cc.get_nmo()
        return int(nmoa), int(nmob)

    @property
    def nvir_ab(self) -> tuple[int, int]:
        """Number of virtual spin-orbitals ``(alpha, beta)``."""
        nocca, noccb = self.nocc
        nmoa, nmob = self.nmo
        return nmoa - nocca, nmob - noccb

    @property
    def nphys(self) -> int:
        """Number of physical (spin-)orbitals (alpha block followed by beta block)."""
        nmoa, nmob = self.nmo
        return nmoa + nmob


class UCCSD_1h(BaseUCCSD):  # pylint: disable=invalid-name
    """IP-EOM-UCCSD expressions."""

    # Hole moments use the left application (only ``ipccsd_matvec`` is available).
    _gf_moments_left = True

    def _precompute_imds(self) -> None:
        """Precompute intermediate integrals."""
        self._imds.make_ip()

    def vector_to_amplitudes(self, vector: Array, *args: Any) -> tuple[Any, Any]:
        """Convert a vector to amplitudes."""
        return self.PYSCF_EOM.vector_to_amplitudes_ip(vector, self.nmo, self.nocc)

    def amplitudes_to_vector(self, r1: Any, r2: Any) -> Array:
        """Convert amplitudes to a vector."""
        return self.PYSCF_EOM.amplitudes_to_vector_ip(r1, r2)

    def apply_hamiltonian_left(self, vector: Array) -> Array:
        """Apply the Hamiltonian to a vector on the left.

        Notes:
            This is backed by PySCF's ``ipccsd_matvec`` (the only available IP matvec). The hole
            Green's-function moments are therefore obtained with ``build_gf_moments(left=True)``,
            which routes the application through this method. See the module docstring.
        """
        return -self.PYSCF_EOM.ipccsd_matvec(self, vector, imds=self._imds)

    def apply_hamiltonian_right(self, vector: Array) -> Array:
        """Apply the Hamiltonian to a vector on the right.

        Raises:
            NotImplementedError: PySCF does not provide ``lipccsd_matvec`` for UCCSD. Use
                ``build_gf_moments(left=True)`` instead, which only needs the left application.
        """
        raise NotImplementedError(
            "PySCF does not provide lipccsd_matvec for UCCSD; use build_gf_moments(left=True)."
        )

    apply_hamiltonian = apply_hamiltonian_left
    apply_hamiltonian.__doc__ = BaseUCCSD.apply_hamiltonian.__doc__

    def diagonal(self) -> Array:
        """Get the diagonal of the Hamiltonian."""
        return -self.PYSCF_EOM.ipccsd_diag(self, imds=self._imds)

    def get_excitation_bra(self, orbital: int) -> Array:
        r"""Obtain the bra excitation vector for a fermionic operator on the ground state."""
        r1, r2 = _ip_bra(self.t1, self.t2, self.l1, self.l2, self.nocc, self.nvir_ab, orbital)
        return self.amplitudes_to_vector(r1, r2)

    def get_excitation_ket(self, orbital: int) -> Array:
        r"""Obtain the ket excitation vector for a fermionic operator on the ground state."""
        r1, r2 = _ip_ket(self.t1, self.t2, self.l1, self.l2, self.nocc, self.nvir_ab, orbital)
        return self.amplitudes_to_vector(r1, r2)

    get_excitation_vector = get_excitation_ket
    get_excitation_vector.__doc__ = BaseUCCSD.get_excitation_vector.__doc__

    @property
    def nsingle(self) -> int:
        """Number of configurations in the singles sector."""
        nocca, noccb = self.nocc
        return nocca + noccb


class UCCSD_1p(BaseUCCSD):  # pylint: disable=invalid-name
    """EA-EOM-UCCSD expressions."""

    # Particle moments use the right application (the default ``eaccsd_matvec``).
    _gf_moments_left = False

    def _precompute_imds(self) -> None:
        """Precompute intermediate integrals."""
        self._imds.make_ea()

    def vector_to_amplitudes(self, vector: Array, *args: Any) -> tuple[Any, Any]:
        """Convert a vector to amplitudes."""
        return self.PYSCF_EOM.vector_to_amplitudes_ea(vector, self.nmo, self.nocc)

    def amplitudes_to_vector(self, r1: Any, r2: Any) -> Array:
        """Convert amplitudes to a vector."""
        return self.PYSCF_EOM.amplitudes_to_vector_ea(r1, r2)

    def apply_hamiltonian_right(self, vector: Array) -> Array:
        """Apply the Hamiltonian to a vector on the right.

        Notes:
            Backed by PySCF's ``eaccsd_matvec``. The particle Green's-function moments are obtained
            with the default ``build_gf_moments(left=False)``.
        """
        return self.PYSCF_EOM.eaccsd_matvec(self, vector, imds=self._imds)

    def apply_hamiltonian_left(self, vector: Array) -> Array:
        """Apply the Hamiltonian to a vector on the left.

        Raises:
            NotImplementedError: PySCF does not provide ``leaccsd_matvec`` for UCCSD. The particle
                moments only need the right application (``build_gf_moments(left=False)``).
        """
        raise NotImplementedError(
            "PySCF does not provide leaccsd_matvec for UCCSD; use build_gf_moments(left=False)."
        )

    apply_hamiltonian = apply_hamiltonian_right
    apply_hamiltonian.__doc__ = BaseUCCSD.apply_hamiltonian.__doc__

    def diagonal(self) -> Array:
        """Get the diagonal of the Hamiltonian."""
        return self.PYSCF_EOM.eaccsd_diag(self, imds=self._imds)

    def get_excitation_bra(self, orbital: int) -> Array:
        r"""Obtain the bra excitation vector for a fermionic operator on the ground state."""
        r1, r2 = _ea_bra(self.t1, self.t2, self.l1, self.l2, self.nocc, self.nvir_ab, orbital)
        return self.amplitudes_to_vector(r1, r2)

    def get_excitation_ket(self, orbital: int) -> Array:
        r"""Obtain the ket excitation vector for a fermionic operator on the ground state."""
        r1, r2 = _ea_ket(self.t1, self.t2, self.l1, self.l2, self.nocc, self.nvir_ab, orbital)
        return self.amplitudes_to_vector(r1, r2)

    get_excitation_vector = get_excitation_ket
    get_excitation_vector.__doc__ = BaseUCCSD.get_excitation_vector.__doc__

    @property
    def nsingle(self) -> int:
        """Number of configurations in the singles sector."""
        nvira, nvirb = self.nvir_ab
        return nvira + nvirb


class UCCSD(ExpressionCollection):
    """Collection of UCCSD expressions for different parts of the Green's function."""

    _hole = UCCSD_1h
    _particle = UCCSD_1p
    _name = "UCCSD"
