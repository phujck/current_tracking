import numpy as np
# import hub_lats as hub
# from pyscf import fci
import evolve
import hams
import hams_track
import hams_verify
import hams_track_branch
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy.integrate import ode
from scipy.interpolate import interp1d
from scipy.signal import fftconvolve
from scipy.signal import blackman
from scipy.signal import stft


def iFT(A):
    """
    Inverse Fourier transform
    :param A:  1D numpy.array
    :return:
    """
    A = np.array(A)
    minus_one = (-1) ** np.arange(A.size)
    result = np.fft.ifft(minus_one * A)
    result *= minus_one
    result *= np.exp(1j * np.pi * A.size / 2)
    return result


def FT(A):
    """
    Fourier transform
    :param A:  1D numpy.array
    :return:
    """
    # test
    A = np.array(A)
    minus_one = (-1) ** np.arange(A.size)
    result = np.fft.fft(minus_one * A)
    result *= minus_one
    result *= np.exp(-1j * np.pi * A.size / 2)
    return result


def wrap(x):
    wrapped = x.copy()

    mod_x1 = np.mod(x, 0.5 * np.pi)
    mod_x2 = np.mod(x, np.pi)
    mod_x3 = np.mod(x, 1.5 * np.pi)

    indx = np.where(x > 0.5 * np.pi)
    wrapped[indx] = 0.5 * np.pi - mod_x1[indx]

    indx = np.where(x < -0.5 * np.pi)
    wrapped[indx] = -mod_x1[indx]

    indx = np.where(x > np.pi)
    wrapped[indx] = -mod_x2[indx]

    indx = np.where(x < - np.pi)
    wrapped[indx] = np.pi - mod_x2[indx]

    indx = np.where(x > 1.5 * np.pi)
    wrapped[indx] = -0.5 * np.pi + mod_x3[indx]

    indx = np.where(x < -1.5 * np.pi)
    wrapped[indx] = - np.pi + mod_x3[indx]

    return wrapped


def progress(total, current):
    if total < 10:
        print("Simulation Progress: " + str(int(round(100 * current / total))) + "%")
    elif current % (total / 10) == 0:
        print("Simulation Progress: " + str(int(round(100 * current / total))) + "%")
    return


nelec = (1, 1)
nx = 8
ny = 0
# U/t is 4.0
U = 4.0 * 0.52
t = 0.52
delta = 1.e-2
cycles = 10

# Tracking time
time1 = 4.54
# Tracking = True
Tracking = True
# Track_Branch = True
Track_Branch = False
Verify = False

prop = hams.system(nelec, nx, ny, U, t, delta, cycles)

# expectations here
up = []
down = []
neighbour = []
phi_original = []
J_field = []
phi_reconstruct = [0, 0]
boundary_1 = []
boundary_2 = []

psi = prop.get_gs()[1].astype(np.complex128)

r = ode(evolve.integrate_f).set_integrator('zvode', method='bdf')
r.set_initial_value(psi, 0).set_f_params(prop)
branch = 0
while r.successful() and r.t < prop.cycles:
    r.integrate(r.t + delta)
    psi = r.y
    time = r.t
    # add to expectations
    progress(prop.n_time, int(time / delta))
    up.append(evolve.spin_up(prop, psi))
    down.append(evolve.spin_down(prop, psi))
    neighbour.append(evolve.nearest_neighbour(prop, psi))
    phi_original.append(prop.phi(time))
    J_field.append(evolve.current(prop, phi_original[-1], neighbour[-1]))
    phi, branch = evolve.phi_reconstruct(prop, J_field[-1], neighbour[-1], phi_reconstruct[-1], phi_reconstruct[-2],
                                         branch)
    phi_reconstruct.append(phi)
    boundary_1.append(evolve.boundary_term_1(prop, psi))
    boundary_2.append(evolve.boundary_term_2(prop, psi))
del phi_reconstruct[0:2]
neighbour = np.array(neighbour)
J_field = np.array(J_field)
phi_original = np.array(phi_original)
phi_reconstruct = np.array(phi_reconstruct)
up = np.array(up)
down = np.array(down)

# Do tracking
if Tracking:
    # # Interpolate J to build a function of t that the ODE solver can use
    # # Need to pad this function as the ODE solver doesn't have the good grace to stop at the end time.
    # padder = 30
    # # Will use this copy of J for the tracking field
    # J_copy = np.pad(J_field, (0, padder), 'constant')
    # # phi_copy = np.pad(phi_original, (0, padder), 'constant')
    # times = np.linspace(0.0, prop.cycles + padder * delta, len(up) + padder)
    times = np.arange(1, len(J_field) + 1) * delta
    print('Begin tracking')
    J_func = interp1d(times, J_field, fill_value=0, bounds_error=False, kind='cubic')

    # Comparing results after changing parameters both with and without tracking. Particularly good comparison is U+10
    # Settings for using tracking:

    prop_track = hams_track.system(nelec, nx, ny, U, t+0.01, delta, cycles, J_func)

    # Settings for direct current comparison:

    # prop_track = hams.system((1,1), nx, ny, U, t, delta, cycles)

    psi = prop_track.get_gs()[1].astype(np.complex128)

    # Use this version for tracking:
    r_track = ode(evolve.integrate_f_track).set_integrator('zvode')

    # And this version for direct comparison
    # r_track = ode(evolve.integrate_f).set_integrator('zvode', method='bdf')
    neighbour_track = []
    phi_track = []
    J_field_track = []
    r_track.set_initial_value(psi, 0).set_f_params(prop_track)

    while r_track.successful() and r_track.t < prop.cycles:
        r_track.integrate(r_track.t + delta)
        psi = r_track.y
        time = r_track.t
        # Add to expectations
        progress(prop_track.n_time, int(time / delta))
        neighbour_track.append(evolve.nearest_neighbour(prop_track, psi))

        # This is required to get the phi that comes out
        # Tracking version
        phi_track.append(prop_track.phi(time, psi))

        # For direct comparison
        # phi_track.append(prop_track.phi(time))

        J_field_track.append(evolve.current(prop_track, phi_track[-1], neighbour_track[-1]))

    J_field_track = np.array(J_field_track)
    phi_track = np.array(phi_track)
    neighbour_track = np.array(neighbour_track)

# Tracking while controlling the branch points
if Track_Branch:
    # # Interpolate J to build a function of t that the ODE solver can use
    # # Need to pad this function as the ODE solver doesn't have the good grace to stop at the end time.
    # padder = 30
    # # Will use this copy of J for the tracking field
    # J_copy = np.pad(J_field, (0, padder), 'constant')
    # # phi_copy = np.pad(phi_original, (0, padder), 'constant')
    # times = np.linspace(0.0, prop.cycles + padder * delta, len(up) + padder)
    times = np.arange(1, len(J_field) + 1) * delta
    print('Begin tracking')
    J_func = interp1d(times, J_field, fill_value=0, bounds_error=False, kind='cubic')

    # Comparing results after changing parameters both with and without tracking. Particularly good comparison is U+10
    # Settings for using tracking:

    prop_track_branch = hams_track_branch.system(nelec, nx, ny, U, t, delta, cycles, J_func)

    # Settings for direct current comparison:

    # prop_track = hams.system((1,1), nx, ny, U, t, delta, cycles)

    psi = prop_track_branch.get_gs()[1].astype(np.complex128)

    # Use this version for tracking:
    r_track = ode(evolve.integrate_f_track_branch).set_integrator('zvode', method='bdf')

    # And this version for direct comparison
    # r_track = ode(evolve.integrate_f).set_integrator('zvode', method='bdf')
    neighbour_track_branch = []
    phi_track_branch = [0, 0]
    J_field_track_branch = []
    branch_track_number = 0
    r_track.set_initial_value(psi, 0).set_f_params(prop_track_branch, phi_track_branch[-1], phi_track_branch[-2],
                                                   branch_track_number)
    # time1 = 1.9* prop.cycles / 4

    time2 = prop.cycles

    while r_track.successful() and r_track.t < time1:
        r_track.integrate(r_track.t + delta)
        psi = r_track.y
        time = r_track.t  # Add to expectations
        progress(prop_track_branch.n_time, int(time / delta))
        neighbour_track_branch.append(evolve.nearest_neighbour(prop_track_branch, psi))
        new_phi_track, branch_track_number = prop_track_branch.phi(time, psi, phi_track_branch[-1],
                                                                   phi_track_branch[-2],
                                                                   branch_track_number)

        #
        # new_phi_track, branch_track_number = prop_track_branch.phi(time, psi, phi_track_branch[-1],
        #                                                        phi_track_branch[-2],
        #                                                        branch_track_number)
        phi_track_branch.append(new_phi_track)

        r_track.set_initial_value(psi, time).set_f_params(prop_track_branch, phi_track_branch[-1], phi_track_branch[-2],
                                                          branch_track_number)

        J_field_track_branch.append(evolve.current(prop_track_branch, phi_track_branch[-1], neighbour_track_branch[-1]))

    prop_track = hams_track.system(nelec, nx, ny, U, t, delta, cycles, J_func)
    r_track = ode(evolve.integrate_f_track).set_integrator('zvode')
    r_track.set_initial_value(psi, time).set_f_params(prop_track)

    while r_track.successful() and r_track.t < time2:
        r_track.integrate(r_track.t + delta)
        psi = r_track.y
        time = r_track.t
        # Add to expectations
        progress(prop_track.n_time, int(time / delta))
        neighbour_track_branch.append(evolve.nearest_neighbour(prop_track, psi))
        phi_track_branch.append(prop_track.phi(time, psi))

        # For direct comparison
        # phi_track.append(prop_track.phi(time))

        J_field_track_branch.append(evolve.current(prop_track, phi_track_branch[-1], neighbour_track_branch[-1]))

    del phi_track_branch[0:2]
    J_field_track_branch = np.array(J_field_track_branch)
    phi_track_branch = np.array(phi_track_branch)
    neighbour_track_branch = np.array(neighbour_track_branch)

if Verify:
    # # Interpolate phi to build a function of t that the ODE solver can use
    # # Need to pad this function as the ODE solver doesn't have the good grace to stop at the end time.
    # padder = 30
    # # Will use this copy of J for the tracking field
    # phi_copy = np.pad(phi_track, (0, padder), 'constant')
    # # phi_copy = np.pad(phi_original, (0, padder), 'constant')
    times = np.arange(1, len(J_field) + 1) * delta
    print('Begin verification')
    # phi_func = interp1d(times, phi_track, fill_value=0, bounds_error=False)
    phi_func = interp1d(times, phi_reconstruct, fill_value=0, bounds_error=False)

    # Comparing results after changing parameters both with and without tracking. Particularly good comparison is U+10
    # Settings for using tracking:

    prop_track = hams_verify.system(nelec, nx, ny, U, t, delta, cycles, phi_func)

    # Settings for direct current comparison:

    # prop_track = hams.system((1,1), nx, ny, U, t, delta, cycles)

    psi = prop_track.get_gs()[1].astype(np.complex128)

    # Use this version for tracking:
    # r_track = ode(evolve.integrate_f_track).set_integrator('zvode', method='bdf')

    # And this version for direct comparison
    r_track = ode(evolve.integrate_f).set_integrator('zvode', method='bdf')

    r_track.set_initial_value(psi, 0).set_f_params(prop_track)
    neighbour_verify = []
    phi_verify = []
    J_field_verify = []

    while r_track.successful() and r_track.t < prop.cycles:
        r_track.integrate(r_track.t + delta)
        psi = r_track.y
        time = r_track.t
        # Add to expectations
        progress(prop_track.n_time, int(time / delta))
        neighbour_verify.append(evolve.nearest_neighbour(prop_track, psi))

        # This is required to get the phi that comes out
        # Tracking version
        # phi_verify.append(prop_track.phi(time, psi))

        # For direct comparison
        phi_verify.append(prop_track.phi(time))

        J_field_verify.append(evolve.current(prop_track, phi_verify[-1], neighbour_verify[-1]))
    neighbour_verify = np.array(neighbour_verify)
    devs = np.sum(J_field - J_field_track)
    print('total deviation of currents between tracking and original %s' % devs.real)

# Plotting
omegas = (np.arange(prop.n_time) - prop.n_time / 2) / (cycles)
t = np.linspace(0.0, prop.cycles, len(up))

plt.plot(t, J_field.real, label='original system')
if Tracking:
    plt.plot(t, J_field_track.real, label='new system (With tracking)', linestyle='dashed')
if Verify:
    plt.plot(t, np.array(J_field_verify).real, label='verification', linestyle='dashdot')
if Track_Branch:
    plt.plot(t, np.array(J_field_track_branch).real, label='New system(with branch tracking)', linestyle='dashdot')
plt.legend()
plt.xlabel('Time [cycles]')
plt.ylabel('current expectation')
plt.show()

# plt.plot(t, up.real)
# plt.plot(t, down.real)
# plt.xlabel('Time [cycles]')
# plt.ylabel('$N$')
# plt.show()


#
# plt.plot(t, neighbour.real, label='real part')
# plt.plot(t, neighbour.imag, label='imag part')
# plt.legend()
# plt.xlabel('Time [cycles]')
# plt.ylabel('nearest-neighbour expectation')
# plt.show()


# plt.plot(t, wrap(phi_original), label='original (wrapped)')
plt.plot(t, phi_original, label='original',linestyle='dashed')
# plt.plot(t, phi_reconstruct, label='Reconstructed', linestyle='dashed')
# plt.plot(t, np.gradient(np.gradient(np.pi -phi_reconstruct,delta),delta)/1000, label='Reconstructed', linestyle='dashed')
if Tracking:
    plt.plot(t, phi_track, label='Tracking', linestyle='dashdot')
if Verify:
    plt.plot(t, phi_verify, label='Verification', linestyle='dotted')
if Track_Branch:
    plt.plot(t, phi_track_branch, label='Tracking with Branches', linestyle='dotted')
plt.plot(t, np.ones(len(t)) * np.pi / 2, color='red')
plt.plot(t, np.ones(len(t)) * -1 * np.pi / 2, color='red')
# plt.plot(t, np.sin(phi_original))
plt.legend()
plt.xlabel('Time [cycles]')
plt.ylabel('$\\phi$')
plt.show()

#Boundary term plots
print(prop.t)
plt.plot(t, boundary_1)
plt.show()

plt.plot(t, boundary_2)
plt.show()

# Comparing current gradient with and without the given expressions.
boundary_1=np.array(boundary_1)
boundary_2=np.array(boundary_2)
diff = phi_original - np.angle(neighbour)
J_grad = -2 * prop.a * prop.t * np.gradient(phi_original) * np.abs(neighbour) * np.cos(diff)
term_2 = prop.a * prop.t * prop.t * (
            np.exp(-1j * 2 * phi_original) * boundary_2 + np.conjugate((np.exp(-1j * 2 * phi_original) * boundary_2)))
term_1= prop.a * prop.t * prop.t * (boundary_1)
plt.plot(t, J_grad + 2 * prop.a * prop.t * (
        np.gradient(np.angle(neighbour)) * np.abs(neighbour) * np.cos(diff) - np.gradient(
    np.abs(neighbour)) * np.sin(diff)), label='gradient calculated via expectations', linestyle='dashdot')
plt.plot(t, J_grad+term_1+term_2, linestyle='dashed',
         label='Gradient using commutators')
plt.plot(t, np.gradient(J_field.real), label='Numerical current gradient')
plt.xlabel('Time [cycles]')
plt.ylabel('$\\dot{J}(t)$')
plt.legend()
plt.show()

plt.plot(t, (np.gradient(np.angle(neighbour)) * np.abs(neighbour) * np.cos(diff) - np.gradient(
    np.abs(neighbour)) * np.sin(diff)) / np.gradient(J_field.real))

plt.show()

plt.plot(t, phi_original - np.angle(neighbour), label='original')
plt.plot(t, phi_reconstruct - np.angle(neighbour), label='Reconstructed', linestyle='dashed')
# plt.plot(t, phi_track - np.angle(neighbour_track), label='Tracking', linestyle='dashdot')
# plt.plot(t, phi_verify - np.angle(neighbour_verify), label='Verification', linestyle='dotted')
if Track_Branch:
    plt.plot(t, phi_track_branch - np.angle(neighbour_track_branch), label='Tracking with Branches', linestyle='dotted')
plt.plot(t, np.ones(len(t)) * np.pi / 2, color='red')
plt.plot(t, np.ones(len(t)) * -1 * np.pi / 2, color='red')
# plt.plot(t, np.sin(phi_original))
plt.legend()
plt.xlabel('Time [cycles]')
plt.ylabel('$\\phi-\delta$')
plt.show()

plt.plot(t, np.abs(neighbour), label='original')
if Tracking:
    plt.plot(t, np.abs(neighbour_track), label='Tracking', linestyle='dashdot')
if Verify:
    plt.plot(t, np.abs(neighbour_verify), label='Verification', linestyle='dotted')
if Track_Branch:
    plt.plot(t, np.abs(neighbour_track_branch), label='Tracking with Branching', linestyle='dotted')
plt.legend()
plt.xlabel('Time [cycles]')
plt.ylabel('$F$')
plt.show()
#
# print('omega size %s' % omegas.size)
# print('blackman window size %s' % blackman(prop.n_time).size)
# print('J field size %s' % J_field.size)
plt.semilogy(omegas, abs(FT(np.gradient(J_field[:prop.n_time]) * blackman(prop.n_time))) ** 2, label='original')
if Tracking:
    plt.semilogy(omegas, abs(FT(np.gradient(J_field_track[:prop.n_time]) * blackman(prop.n_time))) ** 2,
                 label='Tracking')
if Track_Branch:
    plt.semilogy(omegas, abs(FT(np.gradient(J_field_track_branch[:prop.n_time]) * blackman(prop.n_time))) ** 2,
                 label='Tracking With Branches')
plt.legend()
plt.title("output dipole acceleration")
plt.xlim([0, 20])
plt.show()
plt.semilogy(omegas, abs(FT(phi_original[:prop.n_time] * blackman(prop.n_time))) ** 2, label='original')
if Tracking:
    plt.semilogy(omegas, abs(FT(phi_track[:prop.n_time]* blackman(prop.n_time))) ** 2, label='Tracking')
if Track_Branch:
    plt.semilogy(omegas, abs(FT(phi_track_branch[:prop.n_time] * blackman(prop.n_time))) ** 2,
                 label='Tracking With Branch')
plt.legend()
plt.title("input-field")
plt.xlim([0, 20])
plt.show()

Y, X, Z1 = stft(phi_original.real, 1, nperseg=150, window=('gaussian', 2/prop.field))
Z1 = np.abs(Z1) ** 2
# plt.pcolormesh(X*delta, Y*omegas.max()/Y.max(), Z1, norm=colors.LogNorm())
plt.pcolormesh(X * delta, Y * omegas.max() / Y.max(), Z1)

plt.title('STFT Magnitude-Tracking field')
plt.ylim([0, 10])
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time(cycles)')
plt.show()

if Tracking:
    Y, X, Z1 = stft(phi_track.real, 1, nperseg=150, window=('gaussian', 2/(prop.field)))
    Z1 = np.abs(Z1) ** 2
    # plt.pcolormesh(X*delta, Y*omegas.max()/Y.max(), Z1, norm=colors.LogNorm())
    plt.pcolormesh(X * delta, Y * omegas.max() / Y.max(), Z1)

    plt.title('STFT Magnitude-Tracking field')
    plt.ylim([0, 10])
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time(cycles)')
    plt.show()

if Track_Branch:
    Y, X, Z1 = stft((phi_track_branch).real, 1, nperseg=150, window=('gaussian', 2/(prop.field)))
    Z1 = np.abs(Z1) ** 2
    # plt.pcolormesh(X*delta, Y*omegas.max()/Y.max(), Z1, norm=colors.LogNorm())
    plt.pcolormesh(X * delta, Y * omegas.max() / Y.max(), Z1)

    plt.title('STFT Magnitude-Tracking with branch cut')
    plt.ylim([0, 15])
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time(cycles)')
    plt.show()
