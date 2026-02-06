"""Container for an spectral representation (eigenvalues and eigenvectors) of a matrix."""

from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING

from dyson import numpy as np
from dyson import util
from dyson.representations.lehmann import Lehmann
from dyson.representations.representation import BaseRepresentation

if TYPE_CHECKING:
    from dyson.typing import Array


class Spectral(BaseRepresentation):
    r"""Spectral representation matrix with a known number of physical degrees of freedom.

    The eigendecomposition (spectral decomposition) of a matrix consists of the eigenvalues
    :math:`\lambda_k` and eigenvectors :math:`v_{pk}` that represent the matrix as

    .. math::
        \sum_{k} \lambda_k v_{pk} u_{qk}^*,

    where the eigenvectors have right-handed components :math:`v` and left-handed components
    :math:`u`.

    Note that the order of eigenvectors is ``(left, right)``, whilst they act in the order
    ``(right, left)`` in the above equation. The naming convention is chosen to be consistent with
    the eigenvalue decomposition, where :math:`v` may be an eigenvector acting on the right of a
    matrix, and :math:`u` is an eigenvector acting on the left of a matrix.
    """

    def __init__(
        self,
        eigvals: Array,
        eigvecs: Array,
        nphys: int,
        sort: bool = False,
        chempot: float | None = None,
    ):
        """Initialise the object.

        Args:
            eigvals: Eigenvalues of the matrix.
            eigvecs: Eigenvectors of the matrix.
            nphys: Number of physical degrees of freedom.
            sort: Sort the eigenfunctions by eigenvalue.
            chempot: Chemical potential to be used in the Lehmann representations of the self-energy
                and Green's function.
        """
        self._eigvals = eigvals
        self._eigvecs = eigvecs
        self._nphys = nphys
        self.chempot = chempot
        if sort:
            self.sort_()
        if not self.hermitian:
            if eigvecs.ndim != 3:
                raise ValueError(
                    f"Couplings must be 3D for a non-Hermitian system, but got {eigvecs.ndim}D."
                )
            if eigvecs.shape[0] != 2:
                raise ValueError(
                    f"Couplings must have shape (2, nphys, naux) for a non-Hermitian system, "
                    f"but got {eigvecs.shape}."
                )

    @classmethod
    def from_matrix(
        cls, matrix: Array, nphys: int, hermitian: bool = True, chempot: float | None = None
    ) -> Spectral:
        """Create a spectrum from a matrix by diagonalising it.

        Args:
            matrix: Matrix to diagonalise.
            nphys: Number of physical degrees of freedom.
            hermitian: Whether the matrix is Hermitian.
            chempot: Chemical potential to be used in the Lehmann representations of the self-energy
                and Green's function.

        Returns:
            Spectrum object.
        """
        if hermitian:
            eigvals, eigvecs = util.eig(matrix, hermitian=True)
        else:
            eigvals, (left, right) = util.eig_lr(matrix, hermitian=False)
            eigvecs = np.array([left, right])
        return cls(eigvals, eigvecs, nphys, chempot=chempot)

    @classmethod
    def from_self_energy(
        cls,
        static: Array,
        self_energy: Lehmann,
        overlap: Array | None = None,
        sort: bool = False,
    ) -> Spectral:
        """Create a spectrum from a self-energy.

        Args:
            static: Static part of the self-energy.
            self_energy: Self-energy.
            overlap: Overlap matrix for the physical space.
            sort: Sort the eigenfunctions by eigenvalue.

        Returns:
            Spectrum object.
        """
        return cls(
            *self_energy.diagonalise_matrix(static, overlap=overlap),
            self_energy.nphys,
            chempot=self_energy.chempot,
            sort=sort,
        )

    def sort_(self) -> None:
        """Sort the eigenfunctions by eigenvalue.

        Note:
            The object is sorted in place.
        """
        idx = np.argsort(self.eigvals)
        self._eigvals = self.eigvals[idx]
        self._eigvecs = self.eigvecs[..., idx]

    def _get_matrix_block(self, slices: tuple[slice, slice]) -> Array:
        """Get a block of the matrix.

        Args:
            slices: Slices to select.

        Returns:
            Block of the matrix.
        """
        left, right = util.unpack_vectors(self.eigvecs)
        return util.einsum("pk,qk,k->pq", right[slices[0]], left[slices[1]].conj(), self.eigvals)

    def get_static_self_energy(self) -> Array:
        """Get the static part of the self-energy.

        Returns:
            Static self-energy.

        Note:
            The static part of the self-energy is defined as the physical space part of the matrix
            from which the spectrum is derived.
        """
        return self._get_matrix_block((slice(self.nphys), slice(self.nphys)))

    def get_auxiliaries(self) -> tuple[Array, Array]:
        """Get the auxiliary energies and couplings contributing to the dynamic self-energy.

        Returns:
            Auxiliary energies and couplings.

        Note:
            The auxiliary energies are the eigenvalues of the auxiliary subspace, and the couplings
            are the eigenvectors projected back to the auxiliary subspace using the
            physical-auxiliary block of the matrix from which the spectrum is derived.
        """
        phys = slice(None, self.nphys)
        aux = slice(self.nphys, None)

        # Project back to the auxiliary subspace
        subspace = self._get_matrix_block((aux, aux))

        # If there are no auxiliaries, return here
        if subspace.size == 0:
            energies = np.empty((0))
            couplings = np.empty((self.nphys, 0) if self.hermitian else (2, self.nphys, 0))
            return energies, couplings

        # Diagonalise the subspace to get the energies and basis for the couplings
        energies, rotation = util.eig_lr(subspace, hermitian=self.hermitian)

        # Project back to the couplings
        couplings_right = self._get_matrix_block((phys, aux))
        if not self.hermitian:
            couplings_left = self._get_matrix_block((aux, phys)).T.conj()

        # Rotate the couplings to the auxiliary basis
        if self.hermitian:
            couplings = couplings_right @ rotation[0]
        else:
            couplings = np.array([couplings_left @ rotation[0], couplings_right @ rotation[1]])

        return energies, couplings

    def get_dyson_orbitals(self) -> tuple[Array, Array]:
        """Get the Dyson orbitals.

        Returns:
            Dyson orbitals.
        """
        return self.eigvals, self.eigvecs[..., : self.nphys, :]

    def get_overlap(self) -> Array:
        """Get the overlap matrix in the physical space.

        Returns:
            Overlap matrix.

        Note:
            The overlap matrix is defined as the zeroth moment of the Green's function, and is given
            by the inner product of the Dyson orbitals.
        """
        _, orbitals = self.get_dyson_orbitals()
        left, right = util.unpack_vectors(orbitals)
        return util.einsum("pk,qk->pq", right, left.conj())

    def get_self_energy(self, chempot: float | None = None) -> Lehmann:
        """Get the Lehmann representation of the self-energy.

        Args:
            chempot: Chemical potential.

        Returns:
            Lehmann representation of the self-energy.
        """
        if chempot is None:
            chempot = self.chempot
        if chempot is None:
            chempot = 0.0
        return Lehmann(*self.get_auxiliaries(), chempot=chempot)

    def get_greens_function(self, chempot: float | None = None) -> Lehmann:
        """Get the Lehmann representation of the Green's function.

        Args:
            chempot: Chemical potential.

        Returns:
            Lehmann representation of the Green's function.
        """
        if chempot is None:
            chempot = self.chempot
        if chempot is None:
            chempot = 0.0
        return Lehmann(*self.get_dyson_orbitals(), chempot=chempot)

    @classmethod
    def combine_for_greens_function(
        cls,
        *spectrals: Spectral,
        overlap: Array | None = None,
        chempot: float | None = None,
    ) -> Spectral:
        """Combine multiple spectral representations to conserve the Green's functions.

        Args:
            *spectrals: Spectral representations to combine.
            overlap: Overlap matrix for the physical space. If not passed, the overlap matrices from
                the individual spectral representations will be summed.
            chempot: Chemical potential to use for the combined result. If not provided, the
                chemical potential from the individual results will be used, raising an exception
                if they are inconsistent.

        Returns:
            Combined spectral representation.
        """
        return combine_for_greens_function(*spectrals, overlap=overlap, chempot=chempot)

    @classmethod
    def combine_for_self_energy(
        cls,
        *spectrals: Spectral,
        overlap: Array | None = None,
        chempot: float | None = None,
    ) -> Spectral:
        """Combine multiple spectral representations to conserve the self-energies.

        Args:
            *spectrals: Spectral representations to combine.
            overlap: Overlap matrix for the physical space. If not passed, the overlap matrices from
                the individual spectral representations will be summed.
            chempot: Chemical potential to use for the combined result. If not provided, the
                chemical potential from the individual results will be used, raising an exception
                if they are inconsistent.

        Returns:
            Combined spectral representation.
        """
        return combine_for_self_energy(*spectrals, overlap=overlap, chempot=chempot)

    @cached_property
    def overlap(self) -> Array:
        """Get the overlap matrix (the zeroth moment of the Green's function)."""
        orbitals = self.get_dyson_orbitals()[1]
        left, right = util.unpack_vectors(orbitals)
        return util.einsum("pk,qk->pq", right, left.conj())

    @property
    def eigvals(self) -> Array:
        """Get the eigenvalues."""
        return self._eigvals

    @property
    def eigvecs(self) -> Array:
        """Get the eigenvectors."""
        return self._eigvecs

    @property
    def nphys(self) -> int:
        """Get the number of physical degrees of freedom."""
        return self._nphys

    @property
    def neig(self) -> int:
        """Get the number of eigenvalues."""
        return self.eigvals.shape[0]

    @property
    def hermitian(self) -> bool:
        """Check if the spectrum is Hermitian."""
        return self.eigvecs.ndim == 2

    def __eq__(self, other: object) -> bool:
        """Check if two Lehmann representations are equal."""
        if not isinstance(other, Spectral):
            return NotImplemented
        if other.nphys != self.nphys:
            return False
        if other.neig != self.neig:
            return False
        if other.hermitian != self.hermitian:
            return False
        if other.chempot != self.chempot:
            return False
        return np.allclose(other.eigvals, self.eigvals) and np.allclose(other.eigvecs, self.eigvecs)

    def __hash__(self) -> int:
        """Hash the object."""
        return hash((tuple(self.eigvals), tuple(self.eigvecs.flatten()), self.nphys, self.chempot))


def _get_physical_space_size(spectrals: tuple[Spectral, ...]) -> int:
    """Get the size of the physical space from multiple spectral representations.

    Args:
        spectrals: Spectral representations.

    Returns:
        Size of the physical space.

    Raises:
        ValueError: If the physical space sizes are inconsistent.
    """
    nphys_set = {s.nphys for s in spectrals}
    if len(nphys_set) != 1:
        raise ValueError("Inconsistent physical space sizes in spectral representations.")
    return nphys_set.pop()


def _get_chemical_potential(spectrals: tuple[Spectral, ...]) -> float | None:
    """Get the chemical potential from multiple spectral representations.

    Args:
        spectrals: Spectral representations.

    Returns:
        Chemical potential.

    Raises:
        ValueError: If the chemical potentials are inconsistent.
    """
    chempot_set = {s.chempot for s in spectrals if s.chempot is not None}
    if not all(np.isclose(c1, c2) for c1 in chempot_set for c2 in chempot_set):
        raise ValueError("Inconsistent chemical potentials in spectral representations.")
    return chempot_set.pop() if chempot_set else None


def combine_for_greens_function(
    *spectrals: Spectral, overlap: Array | None = None, chempot: float | None = None
) -> Spectral:
    """Combine multiple spectral representations to conserve the Green's functions.

    Args:
        *spectrals: Spectral representations to combine.
        overlap: Overlap matrix for the physical space. If not passed, the overlap matrices from the
            individual spectral representations will be summed.
        chempot: Chemical potential to use for the combined result. If not provided, the chemical
            potential from the individual results will be used, raising an exception if they are
            inconsistent.

    Returns:
        Combined spectral representation.

    Note:
        This function combines the spectral representations in such a way that the resulting Green's
        function is equivalent to the combination of the individual Green's functions. This is not
        guaranteed to conserve the self-energy in the same way.
    """
    if len(spectrals) == 1:
        return spectrals[0]
    nphys = _get_physical_space_size(spectrals)
    hermitian = all(spectral.hermitian for spectral in spectrals)
    if chempot is None:
        chempot = _get_chemical_potential(spectrals)
    if overlap is None:
        overlap = sum((spectral.overlap for spectral in spectrals), np.zeros((nphys, nphys)))

    # Combine the eigenvalues and Dyson orbitals
    eigvals = np.concatenate([spectral.eigvals for spectral in spectrals], axis=0)
    left = np.zeros((nphys, 0))
    right = np.zeros((nphys, 0))
    for spectral in spectrals:
        orbitals = spectral.get_dyson_orbitals()[1]
        l, r = util.unpack_vectors(orbitals)
        left = np.concatenate((left, l), axis=1)
        right = np.concatenate((right, r), axis=1)

    # Get the orthogonalisation matrices
    orth, _ = util.matrix_power(overlap, -0.5, hermitian=hermitian, return_error=False)
    unorth, _ = util.matrix_power(overlap, 0.5, hermitian=hermitian, return_error=False)

    # Find a basis for the null space
    projector = right.T @ left.conj()
    vectors = util.null_space_basis(projector, hermitian=hermitian)

    # Orthogonalise the vectors
    left = left.T @ orth
    right = right.T @ orth.T.conj()

    # Biorthonormalise full set of vectors
    left = np.concatenate([left, vectors[0]], axis=1)
    right = np.concatenate([right, vectors[1]], axis=1)
    left, right = util.biorthonormalise(left, right)

    # Unorthogonalise the vectors
    left[:, :nphys] = left[:, :nphys] @ unorth.T.conj()
    right[:, :nphys] = right[:, :nphys] @ unorth

    # Get the combined result
    eigvecs = np.array([left.T.conj(), right.T.conj()]) if not hermitian else left.T.conj()
    spectral = Spectral(eigvals, eigvecs, nphys, chempot=chempot, sort=True)

    return spectral


def combine_for_self_energy(
    *spectrals: Spectral, overlap: Array | None = None, chempot: float | None = None
) -> Spectral:
    """Combine multiple spectral representations to conserve the self-energies.

    Args:
        *spectrals: Spectral representations to combine.
        overlap: Overlap matrix for the physical space.
        chempot: Chemical potential to use for the combined result. If not provided, the chemical
            potential from the individual results will be used, raising an exception if they are
            inconsistent.

    Returns:
        Combined spectral representation.

    Note:
        This function combines the spectral representations in such a way that the resulting
        self-energy is equivalent to the combination of the individual self-energies. This is not
        guaranteed to conserve the Green's function in the same way.
    """
    if len(spectrals) == 1:
        return spectrals[0]
    nphys = _get_physical_space_size(spectrals)
    if chempot is None:
        chempot = _get_chemical_potential(spectrals)
    if overlap is None:
        overlap = sum((spectral.overlap for spectral in spectrals), np.zeros((nphys, nphys)))

    # Combine the self-energies
    static = sum(
        (spectral.get_static_self_energy() for spectral in spectrals), np.zeros((nphys, nphys))
    )
    self_energy = spectrals[0].get_self_energy(chempot=chempot).copy()
    for spectral in spectrals[1:]:
        self_energy = self_energy.concatenate(spectral.get_self_energy(chempot=chempot))

    # Create the combined spectral representation
    spectral = Spectral.from_self_energy(static, self_energy, overlap=overlap, sort=True)

    return spectral
