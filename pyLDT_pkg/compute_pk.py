import camb
from . import growth_eqns
from . import solve_eqns
import scipy.integrate
import numpy as np

def compute_pk_gr(params, zf):

    ombh2 = params['Omega_b']*(params['h']/100)**2
    omch2 = params['Omega_m']*(params['h']/100)**2 - ombh2

    pars = camb.CAMBparams()
    pars.set_cosmology(H0=params['h'], ombh2=ombh2, omch2=omch2, mnu=0., tau=0.09)
    pars.InitPower.set_params(As=params['A_s'], ns=params['n_s'])
    pars.set_accuracy(AccuracyBoost=1)
    pars.set_matter_power(redshifts=zf, kmax=20, k_per_logint=0, nonlinear=False)
    #Linear power spectrum
    results = camb.get_results(pars)
    return results.get_linear_matter_power_spectrum()


def compute_pk_fr(Omega_m,fR0,n,xf,kvec,pk_gr):
    """Takes GR matter power spectrum pk_gr[ix] at z=zf. Make sure to use corresponding index ix for CAMB output."""

    z_trans = 127
    a_trans = 1/(1+z_trans)
    ai = 1e-5
    growth_lcdm_trans = scipy.integrate.quadrature(growth_eqns.growth_int,1e-10,a_trans,args=(Omega_m,ai,a_trans),tol=1e-8,rtol=1e-8)[0]
    growth_lcdm_fin = scipy.integrate.quadrature(growth_eqns.growth_int,1e-10,np.exp(xf),args=(Omega_m,ai,np.exp(xf)),tol=1e-8,rtol=1e-8)[0]
    D_lcdm = growth_lcdm_trans/growth_lcdm_fin # LCDM growth normalized at a=af

    pk_fr = np.zeros(kvec.shape)    
    for i,k in enumerate(kvec):
        growth_fr = solve_eqns.calc_growth_fr(Omega_m,fR0,n,k,xf)
        D_fr = growth_fr * np.exp(xf)/a_trans
        pk_fr[i] = D_fr**2 * D_lcdm**2 * pk_gr[i]
    
    return pk_fr
    

def compute_pk_fr_vectorized(Omega_m,fR0,n,zvec,kvec,pk_gr):
    """Takes GR matter power spectrum pk_gr at z=0."""

    Nz = len(zvec)
    z_trans = 127
    a_trans = 1/(1+z_trans)
    ai = 1e-10
    growth_lcdm_trans = scipy.integrate.quadrature(growth_eqns.growth_int,ai,a_trans,args=(Omega_m,ai,a_trans),tol=1e-8,rtol=1e-8)[0]
    growth_lcdm_fin = scipy.integrate.quadrature(growth_eqns.growth_int,1e-10,1,args=(Omega_m,ai,1),tol=1e-8,rtol=1e-8)[0]
    D_lcdm = growth_lcdm_trans/growth_lcdm_fin # LCDM growth normalized at a=1
    pk_gr_ini = D_lcdm**2 * pk_gr

    pk_fr = np.repeat(pk_gr_ini, Nz, axis=0) # initialize MG P(k) to GR P(k) at z=z_trans
    for i,k in enumerate(kvec):
        growth_fr = solve_eqns.calc_growth_fr_full(Omega_m,fR0,n,k,0.)
        D_fr = growth_fr(np.log(1/(1+zvec)))[0] * 1/(a_trans*(1+zvec)) # array of Nz values for each k
        pk_fr[:,i] *= D_fr**2
    
    return pk_fr


def compute_pk_dgp_vectorized(Omega_m,rcH0,zvec,pk_gr):
    """Takes GR matter power spectrum pk_gr at z=0."""

    Nz = len(zvec)
    z_trans = 127
    a_trans = 1/(1+z_trans)
    ai = 1e-10
    growth_lcdm_trans = scipy.integrate.quadrature(growth_eqns.growth_int,ai,a_trans,args=(Omega_m,ai,a_trans),tol=1e-8,rtol=1e-8)[0]
    growth_lcdm_fin = scipy.integrate.quadrature(growth_eqns.growth_int,1e-10,1,args=(Omega_m,ai,1),tol=1e-8,rtol=1e-8)[0]
    D_lcdm = growth_lcdm_trans/growth_lcdm_fin # LCDM growth normalized at a=1
    pk_gr_ini = D_lcdm**2 * pk_gr

    pk_dgp = np.repeat(pk_gr_ini, Nz, axis=0) # initialize MG P(k) to GR P(k) at z=z_trans
    growth_dgp = solve_eqns.calc_growth_dgp_full(Omega_m,rcH0,0.)
    D_dgp = growth_dgp(np.log(1/(1+zvec)))[0] * 1/(a_trans*(1+zvec)) # array of Nz values
    D_dgp_mat = np.repeat(D_dgp,len(pk_gr[0])).reshape(pk_dgp.shape) # matrix of Nz * Nk values
    pk_dgp *= D_dgp_mat**2
    
    return pk_dgp