import math
import numpy as np
import scipy
from matplotlib import pyplot as plt


def gamma_func(x):
    return (x ** 2) / (3 * ((1 + x ** 2) ** (1/2)))

def whitedrawf_dydr(t, ystate):

    # you wouldn't think this nessassry in the presence of a zero-checking
    # termination event but it is just trust me
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
    return y[0]
density_zero_event.terminal = True


# initail condition for testing
y0 = np.array([1e-1, 0])
wd_sol = scipy.integrate.solve_ivp(whitedrawf_dydr, t_span=[1e-5, 3e6], y0=y0, dense_output=True, events=density_zero_event)
print(wd_sol)

# plot results
fig1 = plt.figure("figure")
ax1 = fig1.add_subplot()
ax1.set_title("density over r")
# plot each intial condition
ax1.plot(wd_sol.t, wd_sol.y[0], label='density')
ax1.plot(wd_sol.t, wd_sol.y[1], label='mass')
# add labels
#ax1.set_ylabel("Theta (deg)")
#ax1.set_xlabel("Time (s)")
ax1.legend()
plt.show()


