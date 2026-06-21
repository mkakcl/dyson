"""Generalised coupled cluster singles and doubles (GCCSD) expressions.

Spin-orbital (GHF reference) analogue of the restricted CCSD expressions in
:mod:`dyson.expressions.ccsd`. The amplitudes are fully antisymmetric spin-orbital tensors, so the
closed-shell spin-summation factors of the restricted expressions are absent.

.. [1] Purvis, G. D., & Bartlett, R. J. (1982). A full coupled-cluster singles and doubles
   model: The inclusion of disconnected triples. The Journal of Chemical Physics, 76(4), 1910–1918.
   https://doi.org/10.1063/1.443164

.. [2] Stanton, J. F., & Bartlett, R. J. (1993). The equation of motion coupled-cluster
   method. A systematic biorthogonal approach to molecular excitation energies, transition
   probabilities, and excited state properties. The Journal of Chemical Physics, 98(9), 7029–7039.
   https://doi.org/10.1063/1.464746

"""

from __future__ import annotations

import warnings
from abc import abstractmethod
from typing import TYPE_CHECKING

from pyscf import cc, scf

from dyson import numpy as np
from dyson import util
from dyson.expressions.expression import BaseExpression, ExpressionCollection
from dyson.representations.enums import Reduction

if TYPE_CHECKING:
    from typing import Any

    from pyscf.gto.mole import Mole
    from pyscf.scf.hf import SCF

    from dyson.typing import Array


class BaseGCCSD(BaseExpression):
    """Base class for GCCSD expressions.

    Notes:
        Orbital counts (:attr:`nocc`, :attr:`nvir`, :attr:`nphys`) refer to spin-orbitals, taken
        directly from the amplitude shapes rather than from the molecule. This means open-shell
        systems are supported, unlike the restricted expressions.
    """

    hermitian_downfolded = False
    hermitian_upfolded = False

    partition: str | None = None

    PYSCF_EOM = cc.eom_gccsd

    def __init__(
        self,
        mol: Mole,
        t1: Array,
        t2: Array,
        l1: Array,
        l2: Array,
        imds: Any,
    ):
        """Initialise the expression.

        Args:
            mol: Molecule object.
            t1: T1 amplitudes.
            t2: T2 amplitudes.
            l1: L1 amplitudes.
            l2: L2 amplitudes.
            imds: Intermediate integrals.
        """
        self._mol = mol
        self._t1 = t1
        self._t2 = t2
        self._l1 = l1
        self._l2 = l2
        self._imds = imds
        self._precompute_imds()

    @abstractmethod
    def _precompute_imds(self) -> None:
        """Precompute intermediate integrals."""
        pass

    @classmethod
    def from_ccsd(cls, ccsd: cc.gccsd.GCCSD) -> BaseGCCSD:
        """Create an expression from a GCCSD object.

        Args:
            ccsd: GCCSD object.

        Returns:
            Expression object.
        """
        if not ccsd.converged:
            warnings.warn("GCCSD T amplitudes are not converged.", UserWarning, stacklevel=2)
        if not ccsd.converged_lambda:
            warnings.warn("GCCSD L amplitudes are not converged.", UserWarning, stacklevel=2)
        eris = ccsd.ao2mo()
        imds = cls.PYSCF_EOM._IMDS(ccsd, eris=eris)  # pylint: disable=protected-access
        return cls(
            mol=ccsd._scf.mol,  # pylint: disable=protected-access
            t1=ccsd.t1,
            t2=ccsd.t2,
            l1=ccsd.l1,
            l2=ccsd.l2,
            imds=imds,
        )

    @classmethod
    def from_mf(cls, mf: SCF) -> BaseGCCSD:
        """Create an expression from a mean-field object.

        Args:
            mf: Mean-field object. If it is not a generalised (GHF) mean-field, it is converted to
                one via ``mf.to_ghf()``.

        Returns:
            Expression object.
        """
        if not isinstance(mf, scf.ghf.GHF):
            mf = mf.to_ghf()
        ccsd = cc.GCCSD(mf)
        ccsd.conv_tol_normt = 1e-9
        ccsd.kernel()
        ccsd.solve_lambda()
        return cls.from_ccsd(ccsd)

    @abstractmethod
    def vector_to_amplitudes(self, vector: Array, *args: Any) -> tuple[Array, Array]:
        """Convert a vector to amplitudes.

        Args:
            vector: Vector to convert.
            args: Additional arguments, redunantly passed during interoperation with PySCF.

        Returns:
            Amplitudes.
        """
        pass

    @abstractmethod
    def amplitudes_to_vector(self, t1: Array, t2: Array) -> Array:
        """Convert amplitudes to a vector.

        Args:
            t1: T1 amplitudes.
            t2: T2 amplitudes.

        Returns:
            Vector.
        """
        pass

    def build_se_moments(self, nmom: int, reduction: Reduction = Reduction.NONE) -> Array:
        """Build the self-energy moments.

        Args:
            nmom: Number of moments to compute.
            reduction: Reduction method to apply to the moments.

        Returns:
            Moments of the self-energy.
        """
        raise NotImplementedError("Self-energy moments not implemented for GCCSD.")

    @property
    def mol(self) -> Mole:
        """Molecule object."""
        return self._mol

    @property
    def t1(self) -> Array:
        """T1 amplitudes."""
        return self._t1

    @property
    def t2(self) -> Array:
        """T2 amplitudes."""
        return self._t2

    @property
    def l1(self) -> Array:
        """L1 amplitudes."""
        return self._l1

    @property
    def l2(self) -> Array:
        """L2 amplitudes."""
        return self._l2

    @property
    def non_dyson(self) -> bool:
        """Whether the expression produces a non-Dyson Green's function."""
        return False

    # The following properties count spin-orbitals, taken from the amplitude shapes:

    @property
    def nocc(self) -> int:
        """Number of occupied spin-orbitals."""
        return self._t1.shape[0]

    @property
    def nvir(self) -> int:
        """Number of virtual spin-orbitals."""
        return self._t1.shape[1]

    @property
    def nphys(self) -> int:
        """Number of physical (spin-)orbitals."""
        return self.nocc + self.nvir

    @property
    def nmo(self) -> int:
        """Get the number of molecular (spin-)orbitals."""
        return self.nphys


class GCCSD_1h(BaseGCCSD):  # pylint: disable=invalid-name
    """IP-EOM-GCCSD expressions."""

    def _precompute_imds(self) -> None:
        """Precompute intermediate integrals."""
        self._imds.make_ip()

    def vector_to_amplitudes(self, vector: Array, *args: Any) -> tuple[Array, Array]:
        """Convert a vector to amplitudes.

        Args:
            vector: Vector to convert.
            args: Additional arguments, redunantly passed during interoperation with PySCF.

        Returns:
            Amplitudes.
        """
        return self.PYSCF_EOM.vector_to_amplitudes_ip(vector, self.nphys, self.nocc)

    def amplitudes_to_vector(self, t1: Array, t2: Array) -> Array:
        """Convert amplitudes to a vector.

        Args:
            t1: T1 amplitudes.
            t2: T2 amplitudes.

        Returns:
            Vector.
        """
        return self.PYSCF_EOM.amplitudes_to_vector_ip(t1, t2)

    def apply_hamiltonian_right(self, vector: Array) -> Array:
        """Apply the Hamiltonian to a vector on the right.

        Args:
            vector: Vector to apply Hamiltonian to.

        Returns:
            Output vector.

        Notes:
            The Hamiltonian is applied in the opposite direction compared to canonical IP-EOM-CCSD,
            which reflects the opposite ordering of the excitation operators with respect to the
            physical indices in the Green's function. This is only of consequence to non-Hermitian
            Green's functions.
        """
        return -self.PYSCF_EOM.lipccsd_matvec(self, vector, imds=self._imds)

    def apply_hamiltonian_left(self, vector: Array) -> Array:
        """Apply the Hamiltonian to a vector on the left.

        Args:
            vector: Vector to apply Hamiltonian to.

        Returns:
            Output vector.

        Notes:
            The Hamiltonian is applied in the opposite direction compared to canonical IP-EOM-CCSD,
            which reflects the opposite ordering of the excitation operators with respect to the
            physical indices in the Green's function. This is only of consequence to non-Hermitian
            Green's functions.
        """
        return -self.PYSCF_EOM.ipccsd_matvec(self, vector, imds=self._imds)

    apply_hamiltonian = apply_hamiltonian_right
    apply_hamiltonian.__doc__ = BaseGCCSD.apply_hamiltonian.__doc__

    def diagonal(self) -> Array:
        """Get the diagonal of the Hamiltonian.

        Returns:
            Diagonal of the Hamiltonian.
        """
        return -self.PYSCF_EOM.ipccsd_diag(self, imds=self._imds)

    def get_excitation_bra(self, orbital: int) -> Array:
        r"""Obtain the bra vector corresponding to a fermionic operator acting on the ground state.

        Args:
            orbital: Orbital index.

        Returns:
            Bra excitation vector.

        Notes:
            The bra and ket are defined in the opposite direction compared to canonical IP-EOM-CCSD,
            which reflects the opposite ordering of the excitation operators with respect to the
            physical indices in the Green's function.
        """
        r1: Array
        r2: Array

        if orbital < self.nocc:
            r1 = np.eye(self.nocc)[orbital]
            r2 = np.zeros((self.nocc, self.nocc, self.nvir))

        else:
            r1 = self.t1[:, orbital - self.nocc]
            r2 = self.t2[:, :, orbital - self.nocc]

        return self.amplitudes_to_vector(r1, r2)

    def get_excitation_ket(self, orbital: int) -> Array:
        r"""Obtain the ket vector corresponding to a fermionic operator acting on the ground state.

        Args:
            orbital: Orbital index.

        Returns:
            Ket excitation vector.

        Notes:
            The bra and ket are defined in the opposite direction compared to canonical IP-EOM-CCSD,
            which reflects the opposite ordering of the excitation operators with respect to the
            physical indices in the Green's function.
        """
        if orbital < self.nocc:
            r1 = np.eye(self.nocc)[orbital].astype(self.l1.dtype)
            r1 -= util.einsum("ie,e->i", self.l1, self.t1[orbital])
            r1 -= 0.5 * util.einsum("imef,mef->i", self.l2, self.t2[orbital])

            tmp = -0.5 * util.einsum("ijea,e->ija", self.l2, self.t1[orbital])
            r2 = tmp - tmp.swapaxes(0, 1)
            tmp = util.einsum("ja,i->ija", self.l1, np.eye(self.nocc)[orbital])
            r2 += tmp - tmp.swapaxes(0, 1)

        else:
            r1 = self.l1[:, orbital - self.nocc].copy()
            r2 = self.l2[:, :, orbital - self.nocc].copy()

        return self.amplitudes_to_vector(r1, r2)

    get_excitation_vector = get_excitation_ket
    get_excitation_vector.__doc__ = BaseGCCSD.get_excitation_vector.__doc__

    @property
    def nsingle(self) -> int:
        """Number of configurations in the singles sector."""
        return self.nocc

    @property
    def nconfig(self) -> int:
        """Number of configurations."""
        return self.nocc * (self.nocc - 1) // 2 * self.nvir


class GCCSD_1p(BaseGCCSD):  # pylint: disable=invalid-name
    """EA-EOM-GCCSD expressions."""

    def _precompute_imds(self) -> None:
        """Precompute intermediate integrals."""
        self._imds.make_ea()

    def vector_to_amplitudes(self, vector: Array, *args: Any) -> tuple[Array, Array]:
        """Convert a vector to amplitudes.

        Args:
            vector: Vector to convert.
            args: Additional arguments, redunantly passed during interoperation with PySCF.

        Returns:
            Amplitudes.
        """
        return self.PYSCF_EOM.vector_to_amplitudes_ea(vector, self.nphys, self.nocc)

    def amplitudes_to_vector(self, t1: Array, t2: Array) -> Array:
        """Convert amplitudes to a vector.

        Args:
            t1: T1 amplitudes.
            t2: T2 amplitudes.

        Returns:
            Vector.
        """
        return self.PYSCF_EOM.amplitudes_to_vector_ea(t1, t2)

    def apply_hamiltonian_right(self, vector: Array) -> Array:
        """Apply the Hamiltonian to a vector on the right.

        Args:
            vector: Vector to apply Hamiltonian to.

        Returns:
            Output vector.
        """
        return self.PYSCF_EOM.eaccsd_matvec(self, vector, imds=self._imds)

    def apply_hamiltonian_left(self, vector: Array) -> Array:
        """Apply the Hamiltonian to a vector on the left.

        Args:
            vector: Vector to apply Hamiltonian to.

        Returns:
            Output vector.
        """
        return self.PYSCF_EOM.leaccsd_matvec(self, vector, imds=self._imds)

    apply_hamiltonian = apply_hamiltonian_right
    apply_hamiltonian.__doc__ = BaseGCCSD.apply_hamiltonian.__doc__

    def diagonal(self) -> Array:
        """Get the diagonal of the Hamiltonian.

        Returns:
            Diagonal of the Hamiltonian.
        """
        return self.PYSCF_EOM.eaccsd_diag(self, imds=self._imds)

    def get_excitation_bra(self, orbital: int) -> Array:
        r"""Obtain the bra vector corresponding to a fermionic operator acting on the ground state.

        Args:
            orbital: Orbital index.

        Returns:
            Bra excitation vector.
        """
        if orbital < self.nocc:
            r1 = -self.l1[orbital]
            r2 = -self.l2[orbital].copy()

        else:
            r1 = np.eye(self.nvir)[orbital - self.nocc].astype(self.l1.dtype)
            r1 -= util.einsum("mb,m->b", self.l1, self.t1[:, orbital - self.nocc])
            r1 -= 0.5 * util.einsum("kmeb,kme->b", self.l2, self.t2[:, :, :, orbital - self.nocc])

            tmp = -0.5 * util.einsum("ikba,k->iab", self.l2, self.t1[:, orbital - self.nocc])
            r2 = tmp - tmp.swapaxes(1, 2)
            tmp = util.einsum("ib,a->iab", self.l1, np.eye(self.nvir)[orbital - self.nocc])
            r2 += tmp - tmp.swapaxes(1, 2)

        return self.amplitudes_to_vector(r1, r2)

    def get_excitation_ket(self, orbital: int) -> Array:
        r"""Obtain the ket vector corresponding to a fermionic operator acting on the ground state.

        Args:
            orbital: Orbital index.

        Returns:
            Ket excitation vector.
        """
        r1: Array
        r2: Array

        if orbital < self.nocc:
            r1 = self.t1[orbital]
            r2 = self.t2[orbital]

        else:
            r1 = -np.eye(self.nvir)[orbital - self.nocc]
            r2 = np.zeros((self.nocc, self.nvir, self.nvir))

        return -self.amplitudes_to_vector(r1, r2)

    get_excitation_vector = get_excitation_ket
    get_excitation_vector.__doc__ = BaseGCCSD.get_excitation_vector.__doc__

    @property
    def nsingle(self) -> int:
        """Number of configurations in the singles sector."""
        return self.nvir

    @property
    def nconfig(self) -> int:
        """Number of configurations."""
        return self.nocc * self.nvir * (self.nvir - 1) // 2


class GCCSD(ExpressionCollection):
    """Collection of GCCSD expressions for different parts of the Green's function."""

    _hole = GCCSD_1h
    _particle = GCCSD_1p
    _name = "GCCSD"
