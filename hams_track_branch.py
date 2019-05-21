import numpy as np
import hub_lats as hub
from pyscf import fci
import des_cre as dc


class system:
    def __init__(self, nelec, nx, ny, U, t, delta, cycles, J_reconstruct, lat_type='square'):
        self.J_reconstruct = J_reconstruct
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

    def phi(self, current_time, psi, phi_previous_1, phi_previous_2, branch_number):
        # Import the current function
        # if current_time <self.delta:
        #     current=self.J_reconstruct(0)
        # else:
        #     current = self.J_reconstruct(current_time-self.delta)
        current = self.J_reconstruct(current_time)

        # Arrange psi to calculate the nearest neighbour expectations
        psi = np.reshape(psi,
                         (fci.cistring.num_strings(self.nsites, self.nup),
                          fci.cistring.num_strings(self.nsites, self.ndown)))
        D = 0.
        for i in range(self.nsites - 1):
            # add expectation for beta electrons
            D += dc.compute_inner_product(psi, self.nsites, (self.nup, self.ndown), [i, i + 1], [1, 0], [0, 0])
            # add expectation for alpha electrons
            D += dc.compute_inner_product(psi, self.nsites, (self.nup, self.ndown), [i, i + 1], [1, 0], [1, 1])
        # Assuming periodic conditions, add the coupling across the boundary.
        D += dc.compute_inner_product(psi, self.nsites, (self.nup, self.ndown), [self.nsites-1, 0], [1, 0], [0, 0])
        D += dc.compute_inner_product(psi, self.nsites, (self.nup, self.ndown), [self.nsites-1, 0], [1, 0], [1, 1])

        # Use both the current and expectation to construct a field which will reproduce the desired current.
        angle = np.angle(D)
        mag = np.abs(D)
        scalefactor = 2 * self.a * self.t * mag
        # assert np.abs(current)/scalefactor <=1, ('Current too large to reproduce, ration is %s' % np.abs(current/scalefactor))
        arg = -current / (2 * self.a * self.t * mag)
        branch_numbers = [branch_number + k for k in [-1, 0, 1]]
        _, new_branch_number = min(
            (abs((-1) ** l * np.arcsin(arg + 0j) + l * np.pi + angle - 2 * phi_previous_1 + phi_previous_2), l) for l in
            branch_numbers
        )
        # print([(abs((-1) ** l * np.arcsin(arg + 0j) + l * np.pi + angle-phi_previous), l) for l in branch_numbers])
        phi = (-1) ** new_branch_number * np.arcsin(arg + 0j) + new_branch_number * np.pi + angle
        # phi = np.arcsin(arg+0j)
        # phi = np.arcsin(arg+0j) + angle
        # phi = np.arcsin(arg + 0j)
        return phi, new_branch_number

    def apply_hhg_pert(self, current_time, psi, phi_previous_1, phi_previous_2, branch_number):
        if self.field == 0.:
            phi = 0.
        else:
            phi, _ = self.phi(current_time, psi, phi_previous_1, phi_previous_2, branch_number)
            # this uses scaled time: tau=field*t
        return np.exp(1j * phi) * np.triu(self.h1) + np.exp(-1j * phi) * np.tril(self.h1)

    def get_gs(self):

        cisolver = fci.direct_spin1.FCI()
        e, fcivec = cisolver.kernel(self.h1, self.h2, self.lat.nsites, (self.nup, self.ndown))
        return e, fcivec.flatten()
