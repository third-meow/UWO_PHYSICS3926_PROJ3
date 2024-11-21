
# derivative function for pendulum ode, split into 3
def pend_dydt(t,y,Q,A,omega_d):
    theta = y[0]
    phi = y[1]
    omega = y[2]

    dydt = [0, 0, 0]
    dydt[0] = omega
    dydt[1] = omega_d
    dydt[2] = A*np.cos(phi) - (omega/Q + np.sin(theta))
    return dydt

# solve with Q=20, A=0.3, wd=2 from theta_m=30
pend_solution = scipy.integrate.solve_ivp(pend_dydt, [0, 30], [deg_to_rad(30), 0, 0], args=(20,0.3,2))
