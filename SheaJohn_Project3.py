import math
import numpy as np
import scipy
from matplotlib import pyplot as plt

import warnings
warnings.filterwarnings("error")


# constants
mu_e = 2
r0 = 7.72e8 / mu_e
M0 = 5.67e33 / (mu_e ** 2)
rho0 = 9.74e5 * mu_e

mass_of_sun_in_grams = 1.989e33
radius_of_the_sun_in_cm = 6.95508e10

def gamma_func(x):
    """
    Computes the gamma function of x
    """
    return (x ** 2) / (3 * ((1 + x ** 2) ** (1/2)))

def whitedrawf_dydr(t, ystate):
    """
    Calculates the radius-derivative of white dwarf density and mass
    Inputs:
    t - radius in dimensionless units
    ystate - vector containing density and mass at current radius in dimensionless units
    Outputs:
    dydr - vector containing the radius-derivative of density and mass at current radius

    """
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

# checks for density going to zero and terminates solve_ivp process
def density_zero_event(t, y):
    """
    Simply reutrns the first element of y, corresponding to density
    Allows scipy.solve_ivp to stop when density hits zero
    """
    return y[0]
# set terminal = True to tell solve_ivp this is a terminal event
density_zero_event.terminal = True


if __name__ == '__main__':
    """
    Main Script
    """

    # initial rho values
    res_n = 10
    # rho c is 10 values in the range [1e-1, 2.5e6] on a logscale since the smaller values produce more interesting results
    rho_c = np.logspace(np.log10(1e-1), np.log10(2.5e6), res_n)

    # to store mass and radius results for respective initial rho values
    final_mass_bold = np.zeros(res_n)
    final_rad_bold = np.zeros(res_n)

    for i in range(10):
        # run solve_ivp with defualt method, starting with density in rho_c
        wd_sol = scipy.integrate.solve_ivp(
                whitedrawf_dydr, 
                t_span=[1e-10, 3e6], 
                y0=[rho_c[i], 0], 
                dense_output=True, 
                events=density_zero_event)

        # store results
        final_mass_bold[i] = wd_sol.y[1][-1]
        final_rad_bold[i] = wd_sol.t[-1]

    # adjust out of the dimensionless values
    final_mass = final_mass_bold * M0
    final_rad = final_rad_bold * r0

    # plot radius against mass
    fig1 = plt.figure(f"figure")
    ax1 = fig1.add_subplot()
    ax1.set_title("Radius by Mass")
    ax1.scatter(final_mass/mass_of_sun_in_grams, final_rad/radius_of_the_sun_in_cm, label='Simulated')
    ax1.set_xlabel("Mass (solar masses)")
    ax1.set_ylabel("Radius (solar radii)")
    plt.show()

    ## PART 2 Compare Chandrasekhar Limits
    # estimate Chandrasekhar limit 
    estimated_Mch_g = final_mass[-1]
    estimated_Mch_MoS = final_mass[-1] / mass_of_sun_in_grams
    # print results
    print("\nPART 2 - Estimating Chandrasekhar Limit")
    print(f"Estimated : {estimated_Mch_MoS*(mu_e**2)} / (μe)^2 (solar masses)")
    print("Kippenhahn & Weigert (1990) cite Chandrasekhar Limit as 5.836/(μe)^2  (solar masses)")


    ## PART 3 Compare integration methods

    # to store results using DOP853
    DOP853_final_mass_bold = np.zeros(3)
    DOP853_final_rad_bold = np.zeros(3)

    for i in range(3):
        # run solver with the DOP853 method, using the first 3 values in rho_c
        wd_sol = scipy.integrate.solve_ivp(
                whitedrawf_dydr, 
                t_span=[1e-10, 3e6], 
                y0=[rho_c[i], 0], 
                method='DOP853',
                dense_output=True, 
                events=density_zero_event)

        # store results
        DOP853_final_mass_bold[i] = wd_sol.y[1][-1]
        DOP853_final_rad_bold[i] = wd_sol.t[-1]

    # convert from dimensionless units
    DOP853_final_mass = DOP853_final_mass_bold * M0
    DOP853_final_rad = DOP853_final_rad_bold * r0

    print("\nPART 3 - Method comparison RK45 vs DOP853")
    # number of sig figs to print
    n_sf = 4
    # print comparision 
    for i in range(3):
        print(f"With rho_c = {rho_c[i]}:")
        print(f"\tRK45 Method produced:\t\tradius={final_rad[i]:.{n_sf}} mass={final_mass[i]:.{n_sf}}")
        print(f"\tDOP853 Method produced:\t\tradius={DOP853_final_rad[i]:.{n_sf}} mass={DOP853_final_mass[i]:.{n_sf}}")

    ## Part 4 - Observed data vs model
    # get observed data from file
    observed = np.genfromtxt('wd_mass_radius.csv', delimiter=',')[1:]

    # plot Radius v Mass again, this time with the observed data and error bars
    fig1 = plt.figure(f"figure")
    ax1 = fig1.add_subplot()
    ax1.set_title("Radius by Mass : Observed and Simulated")
    ax1.scatter(final_mass/mass_of_sun_in_grams, final_rad/radius_of_the_sun_in_cm, label='Simulated')
    ax1.errorbar(observed[:,0], observed[:,2], xerr=observed[:,1], yerr=observed[:,3], fmt='ro', capsize=5, label='Observed')
    ax1.set_xlabel("Mass (solar masses)")
    ax1.set_ylabel("Radius (solar radii)")
    ax1.legend()
    plt.show()





