import numpy as np
import scipy.optimize
import scipy.integrate
import scipy.interpolate
import mcfit

from . import growth_eqns
from . import solve_eqns
from . import compute_pk

def init_pyLDT():
#initialise all ODEs at import. Add here new ODE to initialise and increase N_models accordingly
    N_models = 3
    res = [None for _ in range(N_models)]
    res[0] = solve_eqns.SphEvoGR(0.,0.3,1e-5,0.)
    res[1] = solve_eqns.calc_growth_fr_full(0.3,-1e-5,1.,0.001,0.)
    res[2] = solve_eqns.calc_growth_dgp_full(0.3,0.5,0.)

    for i,out in enumerate(res):
        if i==0 and np.isnan(out):
            print('ODE initialisation {:d} failed. Check growth_eqns.py and/or solve_eqns.py modules for bugs and dependencies.'.format(i))
        elif i>0 and np.any(np.isnan(out.u)):
            print('ODE initialisation {:d} failed. Check growth_eqns.py and/or solve_eqns.py modules for bugs and dependencies.'.format(i))

init_pyLDT()


def get_tau(Omega_m, zf=0):
    
    # Integration interval
    ai = 1e-5
    af = 1/(1+zf)
    xi = np.log(ai)
    xf = np.log(af)

    N_dlin = 100
    N_dlin1 = 50
    N_dlin2 = N_dlin - N_dlin1 
    dlin_min = -5.5
    dlin_max = 1.3
    dlin_vec1 = np.linspace(dlin_min,0,N_dlin1,endpoint=False)
    dlin_vec2 = np.linspace(1e-4,dlin_max,N_dlin2)
    dlin_vec = np.concatenate((dlin_vec1,dlin_vec2))

    D_lambda = scipy.integrate.quadrature(growth_eqns.growth_int,1e-10,af,args=(Omega_m,ai,af),tol=1e-8,rtol=1e-8)[0]
    dini_vec = dlin_vec/D_lambda

    return solve_eqns.calc_tau_gr(dini_vec, dlin_vec, Omega_m, xi, xf)

def compute_sigma2(kvec,Nz,pk):

    sigma2 = [None for _ in range(Nz)]
    for z_ix in range(Nz):
        
        def f1(k): return 9.*pk(k,z_ix)/(4.*np.pi*k*k)

        F = f1(kvec)
        DBT = mcfit.DoubleBessel(kvec,alpha=1,nu=1.5,q=-1,lowring=True)
        Rvec, G = DBT(F,extrap=True)
        sigma2[z_ix] = scipy.interpolate.CubicSpline(Rvec,G/Rvec**3.)
    
    return sigma2

def get_s2_mu(R, sigma2, sigma2_fid, s2_mu_fid):

    s2_mu = np.zeros_like(s2_mu_fid)

    for ix,rth in enumerate(R):
        sigma2_all_z = np.asarray([sig2(rth) for sig2 in sigma2])
        sigma2_fid_all_z = np.asarray([sig2(rth) for sig2 in sigma2_fid])
        s2_mu[ix] = np.log(1 + sigma2_all_z)/np.log(1 + sigma2_fid_all_z) * s2_mu_fid[ix]

    return s2_mu


def sigma2_gr(params, zvec):
    
    Nz = len(zvec)
    # To get P(k) at all redshifts rescale CAMB output at z=0 with LCDM growth (neglects perturbative and background effect from radiation)
    # This approach is more self-consistent with MG rescaling below, but could be avoided if using MGCAMB/EFTCAMB/Hi_CLASS to get P(k) in MG instead
    k_camb, _, pk_camb = compute_pk.compute_pk_gr(params,[0.])
    D0_lambda = scipy.integrate.quadrature(growth_eqns.growth_int,1e-10,1.,args=(params['Omega_m'],1e-10,1.),tol=1e-8,rtol=1e-8)[0]
    Dz_lambda_norm = np.array([scipy.integrate.quadrature(growth_eqns.growth_int,1e-10,1/(1+zf),args=(params['Omega_m'],1e-10,1/(1+zf)),tol=1e-8,rtol=1e-8)[0] for zf in zvec])/D0_lambda
    Dz_reshaped = Dz_lambda_norm.reshape(len(Dz_lambda_norm),1)
    pk_camb_z = np.repeat(Dz_reshaped**2, len(pk_camb[0]), axis=1) * np.repeat(pk_camb, Nz, axis=0)
    logpk_gr = [scipy.interpolate.CubicSpline(np.log(k_camb),np.log(pk_camb_z[i])) for i in range(Nz)]
    
    def pk_gr(k,z_ix):
        return np.exp(logpk_gr[z_ix](np.log(k)))

    k_sample = np.logspace(-4,np.log10(20),num=512)
    sigma2_gr = compute_sigma2(k_sample,Nz,pk_gr)

    return sigma2_gr, k_camb, pk_camb

def sigma2_fr(params, zvec, kvec, pk_gr):
    
    Nz = len(zvec)

    pk_fr_vec = compute_pk.compute_pk_fr_vectorized(params['Omega_m'], params['fR0'], params['n'], zvec, kvec, pk_gr)
    logpk_fr = [scipy.interpolate.CubicSpline(np.log(kvec),np.log(pk_fr_vec[i])) for i in range(Nz)]
    
    def pk_fr(k,z_ix):
        return np.exp(logpk_fr[z_ix](np.log(k)))

    k_sample = np.logspace(-4,np.log10(20),num=512)
    sigma2_fr = compute_sigma2(k_sample,Nz,pk_fr)

    return sigma2_fr

def sigma2_dgp(params, zvec, kvec, pk_gr):
    
    Nz = len(zvec)

    pk_dgp_vec = compute_pk.compute_pk_dgp_vectorized(params['Omega_m'], params['rcH0'], zvec, pk_gr)
    logpk_dgp = [scipy.interpolate.CubicSpline(np.log(kvec),np.log(pk_dgp_vec[i])) for i in range(Nz)]
    
    def pk_dgp(k,z_ix):
        return np.exp(logpk_dgp[z_ix](np.log(k)))

    k_sample = np.logspace(-4,np.log10(20),num=512)
    sigma2_dgp = compute_sigma2(k_sample,Nz,pk_dgp)

    return sigma2_dgp

def init_fid(params, zvec):
   
    Nz = len(zvec)

    # To get P(k) at all redshifts rescale CAMB output at z=0 with LCDM growth (neglects perturbative and background effect from radiation)
    # This approach is more self-consistent with MG rescaling below, but could be avoided if using MGCAMB/EFTCAMB/Hi_CLASS to get P(k) in MG instead
    k_camb, _, pk_camb = compute_pk.compute_pk_gr(params,[0.])
    D0_lambda = scipy.integrate.quadrature(growth_eqns.growth_int,1e-10,1.,args=(params['Omega_m'],1e-10,1.),tol=1e-8,rtol=1e-8)[0]
    Dz_lambda_norm = np.array([scipy.integrate.quadrature(growth_eqns.growth_int,1e-10,1/(1+zf),args=(params['Omega_m'],1e-10,1/(1+zf)),tol=1e-8,rtol=1e-8)[0] for zf in zvec])/D0_lambda
    Dz_reshaped = Dz_lambda_norm.reshape(len(Dz_lambda_norm),1)
    pk_camb_z = np.repeat(Dz_reshaped**2, len(pk_camb[0]), axis=1) * np.repeat(pk_camb, Nz, axis=0)
    logpk_gr = [scipy.interpolate.CubicSpline(np.log(k_camb),np.log(pk_camb_z[i])) for i in range(Nz)]
    #####################################
    
    def pk_gr(k,z_ix):
        return np.exp(logpk_gr[z_ix](np.log(k)))

    k_sample = np.logspace(-4,np.log10(20),num=512)
    sigma2_gr = compute_sigma2(k_sample,Nz,pk_gr)

    return sigma2_gr

def get_pdf(R, z, tau, s2_mu, sigma2):

    rho_vec = np.logspace(-1,1.13,500)

    def get_psi(R, tau, s2_mu, sigma2):
    # Here R, s2_mu and sigma2 refer to particular elements in the corresponding input arrays
        psi_vec = (sigma2(R)/s2_mu) * (tau(rho_vec)**2/(2*sigma2(R*rho_vec**(1/3))))
        return scipy.interpolate.CubicSpline(rho_vec, psi_vec)
    
    psi_mat = [[get_psi(rth, tau, s2_mu[r_ix,z_ix], sigma2[z_ix]) for z_ix in range(len(sigma2))] for r_ix,rth in enumerate(R)]


    def get_pdf_nn(psi):
    # compute non-normalised PDF
        pdf_nn_vec = np.sqrt((psi(rho_vec,2) + psi(rho_vec,1)/rho_vec)/(2*np.pi))*np.exp(-psi(rho_vec))
        return scipy.interpolate.CubicSpline(rho_vec, pdf_nn_vec)
    
    pdf_nn_mat = [[get_pdf_nn(psi_mat[r_ix][z_ix]) for z_ix in range(len(sigma2))] for r_ix in range(len(R))]

    def mean_igr(rho, pdf_nn):
    # non-normalised PDF mean integrand
        return rho * pdf_nn(rho)

    mean_mat = [[scipy.integrate.quadrature(mean_igr,0.1,10,args=(pdf_nn_mat[r_ix][z_ix]),tol=1e-4,rtol=1e-4,maxiter=100)[0] for z_ix in range(len(sigma2))] for r_ix in range(len(R))]
    norm_mat = [[scipy.integrate.quadrature(pdf_nn_mat[r_ix][z_ix],0.1,10,tol=1e-4,rtol=1e-4,maxiter=100)[0] for z_ix in range(len(sigma2))] for r_ix in range(len(R))]

    def pdf(pdf_nn,mean,norm):
        rho_max = 10.
        pdf_vec = pdf_nn(rho_vec[rho_vec<rho_max] * mean/norm) * mean/norm**2
        return scipy.interpolate.CubicSpline(rho_vec[rho_vec<rho_max], pdf_vec)

    pdf_mat = [[pdf(pdf_nn_mat[r_ix][z_ix],mean_mat[r_ix][z_ix],norm_mat[r_ix][z_ix]) for z_ix in range(len(sigma2))] for r_ix in range(len(R))]

    return pdf_mat

def compute_pdf(cosmo_params, z, R, sig2_gr_fid, s2_mu_gr_fid, want_fr=False, want_dgp=False):

    #Compute function tau(rho) at z=0. Neglecting z-dependence only introduces inaccuracies <0.3% up to z=2
    tau = get_tau(cosmo_params['Omega_m'], zf=0)

    # compute linear and non-linear variances
    sig2_gr, k_camb, pk_camb = sigma2_gr(cosmo_params, z)
    s2_mu_gr = get_s2_mu(R, sig2_gr, sig2_gr_fid, s2_mu_gr_fid)
    if want_fr: 
        sig2_fr = sigma2_fr(cosmo_params, z, k_camb, pk_camb)
        s2_mu_fr = get_s2_mu(R, sig2_fr, sig2_gr_fid, s2_mu_gr_fid)
    if want_dgp: 
        sig2_dgp = sigma2_dgp(cosmo_params, z, k_camb, pk_camb)
        s2_mu_dgp = get_s2_mu(R, sig2_dgp, sig2_gr_fid, s2_mu_gr_fid)

    # compute matter PDF
    pdf_gr_mat = get_pdf(R, z, tau, s2_mu_gr, sig2_gr)
    if want_fr: 
        pdf_fr_mat = get_pdf(R, z, tau, s2_mu_fr, sig2_fr)
    else:
        pdf_fr_mat = None
    if want_dgp: 
        pdf_dgp_mat = get_pdf(R, z, tau, s2_mu_dgp, sig2_dgp)
    else:
        pdf_dgp_mat = None

    return {'gr':pdf_gr_mat, 'fr':pdf_fr_mat, 'dgp':pdf_dgp_mat}