import numpy as np
from julia import Main

def E(a, Omega_m):
    return np.sqrt(Omega_m/a**3. + (1.-Omega_m))

def growth_int(a, Omega_m, ai, af):
    return 2.5*Omega_m*E(af,Omega_m)/ai*1/(a*E(a,Omega_m))**3

jul_rhs_growth_fr = Main.eval("""
    function growth_fr(dy,y,p,x)
        Om0, fR0, n, k = p
        Omega_m = Om0/(Om0 + exp(3*x)*(1-Om0))
        Omega_L = 1 - Omega_m
        m2 = (1/2997.92458)^2 * (Om0*exp(-3*x) + 4*(1-Om0))^(n+2)/(abs(fR0)*(n+1)*(4-3*Om0)^(n+1))
        eps = k^2/(3 * (k^2 + m2*exp(2*x)))

        dy[1] = y[2]
        dy[2] = -(2.5 + 1.5*Omega_L)*y[2] - 1.5*(2*Omega_L - Omega_m*eps)*y[1]
    end
    """)

jul_rhs_growth_dgp = Main.eval("""
    function growth_dgp(dy,y,p,x)
        Om0, rcH0 = p
        Omega_m = Om0/(Om0 + exp(3*x)*(1-Om0))
        Omega_L = 1 - Omega_m
        Om3x = Om0 * exp(-3*x)
        beta = 1 + 2*rcH0 * sqrt(Om3x + (1-Om0)) * (1 - Om3x/(2*(Om3x + (1-Om0))))
        eps = 1/(3*beta)

        dy[1] = y[2]
        dy[2] = -(2.5 + 1.5*Omega_L)*y[2] - 1.5*(2*Omega_L - Omega_m*eps)*y[1]
    end
    """)