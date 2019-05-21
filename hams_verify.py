import numpy as np
import hub_lats as hub
from pyscf import fci
import des_cre as dc


class system:
    def __init__(self, nelec, nx, ny, U, t, delta, cycles, phi_reconstruct, lat_type='square'):
        self.phi_reconstruct = phi_reconstruct
        self.lat = hub.Lattice(nx, ny, lat_type)
        self.nsites = self.lat.nsites
        self.nup = nelec[0]
        self.ndown = nelec[1]
        self.ne = self.nup + self.ndown
        # converts to a'.u, which are atomic units but with energy normalised to t, so
        # that 1 hartree=1t. also, hbar=e=m_e=1/4pi*ep_0=1, and c=1/alpha=137
        self.factor = 1. / (t * 0.036749323)
        self.U = U / t
        self.t = 1.
        assert self.nup <= self.nsites, 'Too many ups!'
        assert self.ndown <= self.nsites, 'Too many downs!'
        self.h2 = self.two_elec_ham()
        self.h1 = hub.create_1e_ham(self.lat, True)

        self.delta = delta  # timestep

        # Change this to change perturbation
        # Set perturbation to HHG perturbation
        # Set constants required (in future, this should be a kwargs to __init__
        # input units: THz (field), eV (t, U), MV/cm (peak amplitude), Angstroms (lattice cst)
        field = 32.9  # Field frequency
        F0 = 10.
        a = 7.56
        # a=10
        self.field = field * self.factor * 0.0000241888
        self.a = (a * 1.889726125) / self.factor
        self.F0 = F0 * 1.944689151e-4 * (self.factor ** 2.)
        self.cycles = cycles
        self.n_time = int(self.cycles / self.delta)  # Number of time points in propagation
        # self.full_1e_ham should return a function which gives the 1e hamiltonian + perturbation with the current time as an argument
        self.full_1e_ham = self.apply_hhg_pert

    def two_elec_ham(self):
        h2 = np.zeros((self.lat.nsites, self.lat.nsites, self.lat.nsites, self.lat.nsites))
        for i in range(self.lat.nsites):
            h2[i, i, i, i] = self.U
        return h2

    def phi(self, current_time):
        phi=self.phi_reconstruct(current_time)
        return phi

    def apply_hhg_pert(self, current_time):
        if self.field == 0.:
            phi = 0.
        else:
            phi = self.phi(current_time)
            # this uses scaled time: tau=field*t
        return np.exp(1j * phi) * np.triu(self.h1) + np.exp(-1j * phi) * np.tril(self.h1)

    def get_gs(self):

        cisolver = fci.direct_spin1.FCI()
        e, fcivec = cisolver.kernel(self.h1, self.h2, self.lat.nsites, (self.nup, self.ndown))
        return (e, fcivec.flatten())
