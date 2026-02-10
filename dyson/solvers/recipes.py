"""Recipes."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

from dyson import numpy as np
from dyson import util
from dyson.solvers.static.exact import Exact
from dyson.expressions.hamiltonian import Hamiltonian

if TYPE_CHECKING:
    from dyson.typing import Array
    from dyson.representations.lehmann import Lehmann


def greens_function_from_hamiltonian(
    hamiltonian: Array,
    overlap: Array | None = None,
    hermitian: bool | None = None,
) -> Lehmann:
    """Construct a Green's function from a Hamiltonian, solving the Hamiltonian exactly.

    This is principally used to find the non-interacting Green's function for a given Hamiltonian,
    which when expressed on a grid, is given by

    .. math:: G(z) = (z - H)^{-1}.

    Args:
        hamiltonian: Hamiltonian matrix.
        overlap: Overlap matrix. If ``None``, the identity matrix is used.
        hermitian: Whether the Hamiltonian is Hermitian. If ``None``, this is inferred from the
            Hamiltonian.

    Returns:
        Green's function.
    """
    if hermitian is None:
        hermitian = np.allclose(hamiltonian, hamiltonian.T.conj())
    if overlap is None:
        overlap = bra = ket = np.eye(hamiltonian.shape[-1], dtype=hamiltonian.dtype)
    else:
        bra, ket = util.unpack_vectors(util.matrix_power(overlap, 0.5)[0])
    exp = Hamiltonian(hamiltonian, bra=bra, ket=ket if not hermitian else None)
    solver = Exact.from_expression(exp)
    solver.kernel()
    return solver.result.get_greens_function()
