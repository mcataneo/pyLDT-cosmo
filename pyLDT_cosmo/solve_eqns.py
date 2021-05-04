import numpy as np
import scipy.interpolate
from . import sph_coll_eqns as sc
from . import growth_eqns as growth
from diffeqpy import de

def SphEvoGR(dini,Omega_m,xi,xf):
    yi = [0.,-dini/3.]
    x_eval = (xi,xf)   
    params = [Omega_m,dini,xi]
    ode = de.ODEProblem(sc.jul_rhs_gr, yi, x_eval, params)
    sol = de.solve(ode, save_everystep=False, save_start=False, abstol=1e-6, reltol=1e-6)
    return sol.u[0][0]

def calc_tau_gr(dini_vec,dlin_vec,Omega_m,xi,xf):
    dNL_vec = np.zeros(dlin_vec.shape)
    for j, dini in enumerate(dini_vec):
        yf = SphEvoGR(dini,Omega_m,xi,xf)
        dNL_vec[j] = (1 + dini) / (np.exp(xi)/np.exp(xf) * yf + 1)**3

    return scipy.interpolate.InterpolatedUnivariateSpline(dNL_vec, dlin_vec, k=5) #scipy.interpolate.CubicSpline(dNL_vec,dlin_vec)

def calc_growth_fr_full(Omega_m,fR0,n,k,xf):
    gi = [1.,0.]
    xi = np.log(1e-10)
    x_eval = (xi,xf)   
    params = [Omega_m,fR0,n,k]
    ode_growth = de.ODEProblem(growth.jul_rhs_growth_fr, gi, x_eval, params)
    sol_growth = de.solve(ode_growth, save_everystep=True, abstol=1e-8, reltol=1e-8)
    return sol_growth

def calc_growth_dgp_full(Omega_m,rcH0,xf):
    gi = [1.,0.]
    xi = np.log(1e-10)
    x_eval = (xi,xf)   
    params = [Omega_m,rcH0]
    ode_growth = de.ODEProblem(growth.jul_rhs_growth_dgp, gi, x_eval, params)
    sol_growth = de.solve(ode_growth, save_everystep=True, abstol=1e-8, reltol=1e-8)
    return sol_growth