import math
import numpy as np
import scipy
from matplotlib import pyplot as plt

import warnings
warnings.filterwarnings("error")


# important constants
mu_e = 2
r0 = 7.72e8 / mu_e
M0 = 5.67e33 / (mu_e ** 2)
rho0 = 9.74e5 * mu_e

def gamma_func(x):
    return (x ** 2) / (3 * ((1 + x ** 2) ** (1/2)))

def whitedrawf_dydr(t, ystate):
    # you wouldn't think this nessassry in the presence of a zero-checking
    # termination event but it is just trust me
    try:
        if ystate[0] < 0:
            dydr = [0, 0]
            return dydr

        # give sensible names
        r_bold = t
        rho_bold = ystate[0]
        m_bold = ystate[1]

        # calculate x
        x = rho_bold ** (1/3)

        dydr = [0, 0]

        # calculate derivatives
        dydr[0] = -m_bold*rho_bold/(gamma_func(x)*(r_bold ** 2))
        dydr[1] = (r_bold ** 2) * rho_bold

        # return derivatives array
        return dydr
    except RuntimeWarning:
        print(f"RunTimeWarning w/ ystate = {ystate}, dydr = {dydr}")

# checks for density going to zero and terminates solve_ivp process
def density_zero_event(t, y):
    return y[0]
density_zero_event.terminal = True


if __name__ == '__main__':

    # initial rho values
    res_n = 10
    rho_c = np.logspace(np.log10(1e-1), np.log10(2.5e6), res_n)
    #rho_c = np.linspace(1e-1, 2.5e6, res_n)
    final_mass_bold = np.zeros(res_n)
    final_rad_bold = np.zeros(res_n)

    for i in range(10):
        wd_sol = scipy.integrate.solve_ivp(
                whitedrawf_dydr, 
                t_span=[1e-10, 3e6], 
                y0=[rho_c[i], 0], 
                dense_output=True, 
                events=density_zero_event)

        final_mass_bold[i] = wd_sol.y[1][-1]
        final_rad_bold[i] = wd_sol.t[-1]

    final_mass = final_mass_bold * M0
    final_rad = final_rad_bold * r0

    fig1 = plt.figure(f"figure")
    ax1 = fig1.add_subplot()
    ax1.set_title("Radius by Mass")
    # plot each intial condition
    ax1.scatter(final_mass, final_rad, label='not usre')
    ax1.set_xlabel("Mass (g)")
    ax1.set_ylabel("Radius (cm)")
    plt.show()

    ## PART 2 Compare Chandrasekhar Limits
    mass_of_sun_in_grams = 1.989e33
    radius_of_the_sun_in_cm = 6.95508e10
    estimated_Mch_g = final_mass[-1]
    estimated_Mch_MoS = final_mass[-1] / mass_of_sun_in_grams
    print(f"Estimated Chandrasekhar Limit : {estimated_Mch_MoS*(mu_e**2)} / (μe)^2 (solar masses)")
    print("Kippenhahn & Weigert (1990) cite Chandrasekhar Limit as 5.836/(μe)^2  (solar masses)")


    ## PART 3 Compare intergration methods

    DOP853_final_mass_bold = np.zeros(3)
    DOP853_final_rad_bold = np.zeros(3)

    for i in range(3):
        wd_sol = scipy.integrate.solve_ivp(
                whitedrawf_dydr, 
                t_span=[1e-10, 3e6], 
                y0=[rho_c[i], 0], 
                method='DOP853',
                dense_output=True, 
                events=density_zero_event)

        DOP853_final_mass_bold[i] = wd_sol.y[1][-1]
        DOP853_final_rad_bold[i] = wd_sol.t[-1]

    DOP853_final_mass = DOP853_final_mass_bold * M0
    DOP853_final_rad = DOP853_final_rad_bold * r0

    print(DOP853_final_rad)
    print(DOP853_final_mass)
    print(final_mass[:3])
    print(final_rad[:3])

    # number of sig figs to print
    n_sf = 4
    for i in range(3):
        print(f"With rho_c = {rho_c[i]}:")
        print(f"RK45 Method produced:\t\tradius={final_rad[i]:.{n_sf}} mass={final_mass[i]:.{n_sf}}")
        print(f"DOP853 Method produced:\t\tradius={DOP853_final_rad[i]:.{n_sf}} mass={DOP853_final_mass[i]:.{n_sf}}")

    ## Part 4 - Observed data vs model

    observed = np.genfromtxt('wd_mass_radius.csv', delimiter=',')[1:]
    fig1 = plt.figure(f"figure")
    ax1 = fig1.add_subplot()
    ax1.set_title("Radius by Mass : Observed and Simulated")
    # plot each initial condition
    ax1.scatter(final_mass/mass_of_sun_in_grams, final_rad/radius_of_the_sun_in_cm, label='Simulated')

    ax1.errorbar(observed[:,0], observed[:,2], xerr=observed[:,1], yerr=observed[:,3], fmt='ro', capsize=5, label='Observed')

    ax1.set_xlabel("Mass (solar masses)")
    ax1.set_ylabel("Radius (solar radii)")
    ax1.legend()
    plt.show()





