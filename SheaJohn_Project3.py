

def gamma_func(x):
    return (x ** 2) / (3 * ((1 + x ** 2) ** (1/2)))

def whitedrawf_dydr(r_bold,ystate):
    rho_bold = ystate[0]
    m_bold = ystate[1]
    x = rho_bold ** (1/3)

    dydr = [0, 0]
    dydr[0] = -m_bold*rho_bold/(gamma(x)*r*r)
    dydr[1] = r*r*rho_bold
    return dydr

# solve with Q=20, A=0.3, wd=2 from theta_m=30
pend_solution = scipy.integrate.solve_ivp(pend_dydt, [0, 30], [deg_to_rad(30), 0, 0], args=(20,0.3,2))
