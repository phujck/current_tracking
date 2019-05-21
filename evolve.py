import numpy as np
import hub_lats as hub
from pyscf import fci
import des_cre as dc


def apply_H(sys, h1, psi):
    ''' Apply hamiltonian to arbitrary wavefunction, using given h1 + sys.h2. This is the hubbard hamiltonian + perturbation.
        Real and imaginary parts of wavefunction are dealt with separately'''

    psi_r = psi.real
    psi_i = psi.imag
    h1_r = h1.real
    h1_i = h1.imag
    # H|psi>=(h1+h2)|psi>=(h1_r+ih1_i+h2)|psi>=(h1_r+ih1_i+h2)|psi_r>+i(h1_r+ih1_i+h2)|psi_i>
    if sys.U > 0.:
        pro = one_elec(sys, h1_r, psi_r) + 1j * one_elec(sys, h1_i, psi_r, False) \
              + 1j * one_elec(sys, h1_r, psi_i) - one_elec(sys, h1_i, psi_i, False) + two_elec(sys, psi_r, psi_i)
    else:
        pro = one_elec(sys, h1_r, psi_r) + 1j * one_elec(sys, h1_r, psi_i) + two_elec(sys, psi_r, psi_i)
    return pro.flatten()


def one_elec(sys, h1, psi, sym=True):
    ''' Apply one-electron hamiltonian, h1
        sym tells us whether this is a symmetric (real) hamiltonian or not'''
    if sym:
        return fci.direct_spin1.contract_1e(h1, psi, sys.nsites, (sys.nup, sys.ndown))
    else:
        return fci.direct_nosym.contract_1e(h1, psi, sys.nsites, (sys.nup, sys.ndown))


def two_elec(sys, psi_r, psi_i):
    ''' Apply 2-electron hubbard hamiltonian'''
    pro = 0.5 * fci.direct_uhf.contract_2e_hubbard((0, sys.U, 0), psi_r, sys.nsites, (sys.nup, sys.ndown)) \
          + 0.5 * 1j * fci.direct_uhf.contract_2e_hubbard((0, sys.U, 0), psi_i, sys.nsites, (sys.nup, sys.ndown))
    return pro.flatten()


def RK4(sys, current_time, psi):
    '''RK4.
    sys.delta/sys.field scales time by number of cycles'''

    h1_k1 = sys.full_1e_ham(current_time)
    k1 = (-1j * sys.delta / sys.field) * apply_H(sys, h1_k1, psi)

    h1_k2 = sys.full_1e_ham(current_time + 0.5 * sys.delta)
    k2 = (-1j * sys.delta / sys.field) * apply_H(sys, h1_k2, psi + 0.5 * k1)

    k3 = (-1j * sys.delta / sys.field) * apply_H(sys, h1_k2, psi + 0.5 * k2)

    h1_k4 = sys.full_1e_ham(current_time + sys.delta)
    k4 = (-1j * sys.delta / sys.field) * apply_H(sys, h1_k1, psi + k3)

    return psi + (k1 + 2. * k2 + 2. * k3 + k4) / 6.


def RK1(sys, current_time, psi):
    '''Euler step for time evolution
    sys.delta/sys.field scales time by number of cycles'''

    h1 = sys.full_1e_ham(sys, current_time)
    k1 = (-1j * sys.delta / sys.field) * apply_H(sys, h1, psi)
    return psi + k1


def spin_up(sys, psi):
    '''Calculate double occupancy of wavefunction'''
    psi = np.reshape(psi,
                     (fci.cistring.num_strings(sys.nsites, sys.nup), fci.cistring.num_strings(sys.nsites, sys.ndown)))
    D = 0.
    for i in range(sys.nsites):
        # D += dc.compute_inner_product(psi,sys.nsites,(sys.nup,sys.ndown),[i,i,i,i],[1,0,1,0],[1,1,0,0])
        D += dc.compute_inner_product(psi, sys.nsites, (sys.nup, sys.ndown), [i, i], [1, 0], [1, 1])
    return D


def spin_down(sys, psi):
    '''Calculate double occupancy of wavefunction'''
    psi = np.reshape(psi,
                     (fci.cistring.num_strings(sys.nsites, sys.nup), fci.cistring.num_strings(sys.nsites, sys.ndown)))
    D = 0.
    for i in range(sys.nsites):
        # D += dc.compute_inner_product(psi,sys.nsites,(sys.nup,sys.ndown),[i,i,i,i],[1,0,1,0],[1,1,0,0])
        D += dc.compute_inner_product(psi, sys.nsites, (sys.nup, sys.ndown), [i, i], [1, 0], [0, 0])
    return D


def nearest_neighbour(sys, psi):
    "Calculate the nearest neighbour expectation (over both spins)"
    psi = np.reshape(psi,
                     (fci.cistring.num_strings(sys.nsites, sys.nup), fci.cistring.num_strings(sys.nsites, sys.ndown)))
    D = 0.
    for i in range(sys.nsites-1):
        # add expectation for beta electrons
        D += dc.compute_inner_product(psi, sys.nsites, (sys.nup, sys.ndown), [i, i + 1], [1, 0], [0, 0])
        # add expectation for alpha electrons
        D += dc.compute_inner_product(psi, sys.nsites, (sys.nup, sys.ndown), [i, i + 1], [1, 0], [1, 1])
    # Assuming periodic conditions, add the coupling across the boundary.
    D += dc.compute_inner_product(psi, sys.nsites, (sys.nup, sys.ndown), [sys.nsites-1, 0], [1, 0], [0, 0])
    D += dc.compute_inner_product(psi, sys.nsites, (sys.nup, sys.ndown), [sys.nsites-1, 0], [1, 0], [1, 1])
    return D

def boundary_term_1(sys, psi):
    "Calculate the nearest neighbour expectation (over both spins)"
    D=0.
    for i in range(sys.nsites-1):
        # add expectation for beta electrons
        D += dc.compute_inner_product(psi, sys.nsites, (sys.nup, sys.ndown), [i+1, i + 1], [1, 0], [0, 0])
        # add expectation for alpha electrons
        D += dc.compute_inner_product(psi, sys.nsites, (sys.nup, sys.ndown), [i+1, i + 1], [1, 0], [1, 1])
        D -= dc.compute_inner_product(psi, sys.nsites, (sys.nup, sys.ndown), [i, i], [1, 0], [0, 0])
        # add expectation for alpha electrons
        D -= dc.compute_inner_product(psi, sys.nsites, (sys.nup, sys.ndown), [i, i], [1, 0], [1, 1])

    D += dc.compute_inner_product(psi, sys.nsites, (sys.nup, sys.ndown), [0, 0], [1, 0], [0, 0])
    # add expectation for alpha electrons
    D += dc.compute_inner_product(psi, sys.nsites, (sys.nup, sys.ndown), [0, 0], [1, 0], [1, 1])
    D -= dc.compute_inner_product(psi, sys.nsites, (sys.nup, sys.ndown), [sys.nsites-1, sys.nsites-1], [1, 0], [0, 0])
    # add expectation for alpha electrons
    D -= dc.compute_inner_product(psi, sys.nsites, (sys.nup, sys.ndown), [sys.nsites-1, sys.nsites-1], [1, 0], [1, 1])
    return D

def boundary_term_2(sys, psi):
    "Calculate the nearest neighbour expectation (over both spins)"
    D=0.
    D += dc.compute_inner_product(psi, sys.nsites, (sys.nup, sys.ndown), [sys.nsites-1,1], [1, 0], [0, 0])
    # add expectation for alpha electrons
    D += dc.compute_inner_product(psi, sys.nsites, (sys.nup, sys.ndown), [sys.nsites-1, 1], [1, 0], [1, 1])
    D -= dc.compute_inner_product(psi, sys.nsites, (sys.nup, sys.ndown), [0, 2], [1, 0], [0, 0])
    # add expectation for alpha electrons
    D -= dc.compute_inner_product(psi, sys.nsites, (sys.nup, sys.ndown), [0, 2], [1, 0], [1, 1])
    for i in range(1,sys.nsites-2):
        # add expectation for beta electrons
        D += dc.compute_inner_product(psi, sys.nsites, (sys.nup, sys.ndown), [i-1, i+ 1], [1, 0], [0, 0])
        # add expectation for alpha electrons
        D += dc.compute_inner_product(psi, sys.nsites, (sys.nup, sys.ndown), [i-1, i+1], [1, 0], [1, 1])
        D -= dc.compute_inner_product(psi, sys.nsites, (sys.nup, sys.ndown), [i, i+2], [1, 0], [0, 0])
        # add expectation for alpha electrons
        D -= dc.compute_inner_product(psi, sys.nsites, (sys.nup, sys.ndown), [i, i+2], [1, 0], [1, 1])

    D += dc.compute_inner_product(psi, sys.nsites, (sys.nup, sys.ndown), [sys.nsites-3, sys.nsites-1], [1, 0], [0, 0])
    # add expectation for alpha electrons
    D += dc.compute_inner_product(psi, sys.nsites, (sys.nup, sys.ndown), [sys.nsites-3, sys.nsites-1], [1, 0], [1, 1])

    D += dc.compute_inner_product(psi, sys.nsites, (sys.nup, sys.ndown), [sys.nsites-2, 0], [1, 0], [0, 0])
    # add expectation for alpha electrons
    D += dc.compute_inner_product(psi, sys.nsites, (sys.nup, sys.ndown), [sys.nsites-2, 0], [1, 0], [1, 1])

    D -= dc.compute_inner_product(psi, sys.nsites, (sys.nup, sys.ndown), [sys.nsites-2, 0], [1, 0], [0, 0])
    # add expectation for alpha electrons
    D -= dc.compute_inner_product(psi, sys.nsites, (sys.nup, sys.ndown), [sys.nsites-2, 0], [1, 0], [1, 1])

    D -= dc.compute_inner_product(psi, sys.nsites, (sys.nup, sys.ndown), [sys.nsites-1, 1], [1, 0], [0, 0])
    # add expectation for alpha electrons
    D -= dc.compute_inner_product(psi, sys.nsites, (sys.nup, sys.ndown), [sys.nsites-1, 1], [1, 0], [1, 1])
    return D




def current(sys, phi, neighbour):
    conjugator = np.exp(-1j * phi) * neighbour
    c = -1j * sys.a * sys.t * (conjugator - np.conj(conjugator))
    return c


def phi_reconstruct(sys, current, neighbourexpect, phi_previous_1,phi_previous_2, branch_number):
    angle = np.angle(neighbourexpect)
    mag = np.abs(neighbourexpect)
    arg = -current / (2 * sys.a * sys.t * mag)
    branch_numbers=[branch_number+k for k in[-1,0,1]]
    _, new_branch_number = min(
        (abs((-1) ** l * np.arcsin(arg + 0j) + l * np.pi + angle-2*phi_previous_1+phi_previous_2), l) for l in branch_numbers
    )
    # print([(abs((-1) ** l * np.arcsin(arg + 0j) + l * np.pi + angle-phi_previous), l) for l in branch_numbers])
    new_phi=(-1) ** new_branch_number * np.arcsin(arg + 0j) + new_branch_number* np.pi + angle
    # phi = np.arcsin(arg+0j)
    return new_phi, new_branch_number


def integrate_f(t, psi, sys):
    return -1j * apply_H(sys, sys.full_1e_ham(t), psi)


def integrate_f_track(t, psi, sys):
    return -1j * apply_H(sys, sys.full_1e_ham(t, psi), psi)


def integrate_f_track_branch(t, psi, sys, phi_previous_1, phi_previous_2, branch_number):
    return -1j * apply_H(sys, sys.full_1e_ham(t, psi, phi_previous_1, phi_previous_2, branch_number), psi)