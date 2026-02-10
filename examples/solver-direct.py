"""Example of the direct frequency-space solver.

This solver is the traditional form of the frequency-dependent Dyson equation, inverting the
downfolded Green's function matrix at each frequency point.
"""

import numpy
from pyscf import gto, scf

from dyson import FCI, Direct, Exact
from dyson.grids import RealFrequencyGrid
from dyson.representations.spectral import Spectral
from dyson.solvers.recipes import greens_function_from_hamiltonian

# Get a molecule and mean-field from PySCF
mol = gto.M(atom="Li 0 0 0; H 0 0 1.64", basis="sto-3g", verbose=0)
mf = scf.RHF(mol)
mf.kernel()

# Use an FCI expression for the Hamiltonian
exp_h = FCI.hole.from_mf(mf)
exp_p = FCI.particle.from_mf(mf)

# Initialise a real frequency grid for the direct solver
grid = RealFrequencyGrid.from_uniform(-5.0, 5.0, 512, eta=1e-2)

# Use the exact solver to get the central self-energy for demonstration purposes
exact_h = Exact.from_expression(exp_h)
exact_h.kernel()
exact_p = Exact.from_expression(exp_p)
exact_p.kernel()
assert exact_h.result is not None and exact_p.result is not None
result = Spectral.combine_for_self_energy(exact_h.result, exact_p.result)
static = result.get_static_self_energy()
self_energy = result.get_self_energy()
overlap = result.get_overlap()

# Solve the Hamiltonian using the Direct solver, initialisation via either:

# 1) Create the solver from a self-energy. 
#    
#    Note that the default time-ordering (ordered) will display a warning as it is numerically
#    unstable, but the default is kept consistent with other functionality in the package).
solver = Direct.from_self_energy(
    static, self_energy, overlap=overlap, grid=grid, ordering="advanced"
)
solver.kernel()

# 2) Create the solver directly from the initial Green's function and self-energy
gf_0 = grid.evaluate_lehmann(
    greens_function_from_hamiltonian(static, overlap=overlap),
    ordering="advanced",
)
se = grid.evaluate_lehmann(self_energy, ordering="advanced")
solver = Direct(gf_0, se, overlap=overlap)
solver.kernel()

# 3) Don't bother using the solver at all because this is trivial!
gf = (gf_0.inverse() - se).inverse()

# Compare to that of the Exact solver, by downfolding the Green's function corresponding to the
# exact result onto the same grid
gf_exact = grid.evaluate_lehmann(result.get_greens_function(), ordering="advanced")
print("Direct solver error:", numpy.max(numpy.abs(gf - gf_exact)))
