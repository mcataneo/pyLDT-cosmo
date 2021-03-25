# if running pyLDT from conda environment disable compilation to avoid conflicts
import sys, os 
is_conda = os.path.exists(os.path.join(sys.prefix, 'conda-meta'))
if is_conda:
    from julia.api import Julia
    jl = Julia(compiled_modules=False)

import numpy as np
import scipy.optimize
import scipy.integrate
import scipy.interpolate
import mcfit

from . import growth_eqns
from . import solve_eqns
from . import compute_pk


def init_ode():
    """
    Initialise all ODEs when loading pyLDT. Add here new ODE to initialise and increase N_models accordingly
    """

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

init_ode()


class matter_pdf:

    def __init__(self, params, zvec, Rvec, s2_mu_fid):
        """
        Create new instance of matter_pdf class with user-defined fiducial cosmology, redshifts, smoothing radii
        """

        self.z = np.asarray(zvec, dtype=np.float64)
        self.R = np.asarray(Rvec, dtype=np.float64)

        Nz = len(zvec)

        # To get P(k) at all redshifts rescale CAMB output at z=0 with LCDM growth (neglects perturbative and background effect from radiation)
        # This approach is more self-consistent with MG rescaling below, but could be avoided if using MGCAMB/EFTCAMB/Hi_CLASS to get P(k) in MG instead
        k_camb, _, pk_camb = compute_pk.compute_pk_gr(params,[0.])
        D0_lambda = scipy.integrate.quadrature(growth_eqns.growth_int,1e-10,1.,args=(params['Omega_m'],1e-10,1.),tol=1e-8,rtol=1e-8)[0]
        Dz_lambda_norm = np.array([scipy.integrate.quadrature(growth_eqns.growth_int,1e-10,1/(1+zf),args=(params['Omega_m'],1e-10,1/(1+zf)),tol=1e-8,rtol=1e-8)[0] for zf in zvec])/D0_lambda
        Dz_reshaped = Dz_lambda_norm.reshape(len(Dz_lambda_norm),1)
        pk_camb_z = np.repeat(Dz_reshaped**2, len(pk_camb[0]), axis=1) * np.repeat(pk_camb, Nz, axis=0)
        logpk = [scipy.interpolate.CubicSpline(np.log(k_camb),np.log(pk_camb_z[i])) for i in range(Nz)]
        #####################################

        def pk(k,z_ix):
            return np.exp(logpk[z_ix](np.log(k)))

        k_sample = np.logspace(-4,np.log10(20),num=512)
        self.sigma2_fid = self.compute_sigma2(k_sample,Nz,pk)

        self.s2_mu_fid = s2_mu_fid
        self.tau = self.get_tau(1., zf=0) # fix spherical evolution to EdS; sensitivity to DE/MG is negligible anyway


    def get_tau(self, Omega_m, zf=0):
        """
        Returns spline object to the linear density field tau as a function of the non-linear density rho computed in LCDM. Takes
        background total matter density and final redshift as input parameters.
        """

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


    def compute_sigma2(self, kvec, Nz, pk):
        """
        For a given linear power spectrum P(k,z), returns spline object to sigma^2(R) computed 
        with the Hankel Transform.
        """

        sigma2 = [None for _ in range(Nz)]
        for z_ix in range(Nz):
            
            def f1(k): return 9.*pk(k,z_ix)/(4.*np.pi*k*k)

            F = f1(kvec)
            DBT = mcfit.DoubleBessel(kvec,alpha=1,nu=1.5,q=-1,lowring=True)
            Rvec, G = DBT(F,extrap=True)
            sigma2[z_ix] = scipy.interpolate.CubicSpline(Rvec,G/Rvec**3.)

        return sigma2


    def get_s2_mu(self, sigma2):
        """
        Uses the log-normal approximation to get non-linear variance of log-density field from 
        that of the fiducial input cosmology
        """

        s2_mu = np.zeros_like(self.s2_mu_fid)

        for ix,rth in enumerate(self.R):
            sigma2_all_z = np.asarray([sig2(rth) for sig2 in sigma2])
            sigma2_fid_all_z = np.asarray([sig2(rth) for sig2 in self.sigma2_fid])
            s2_mu[ix] = np.log(1 + sigma2_all_z)/np.log(1 + sigma2_fid_all_z) * self.s2_mu_fid[ix]
        
        return s2_mu
    

    def sigma2_gr(self, params):
        """
        Compute a list of length Nz of sigma_lin^2(R,z[i]) functions for GR given an input cosmology and 
        also returns the z=0 GR linear matter power spectrum and wavenumbers. 
        """

        Nz = len(self.z)
        # To get P(k) at all redshifts rescale CAMB output at z=0 with LCDM growth (i.e. neglects perturbative and background effect from radiation)
        # This approach is more self-consistent with MG rescaling below, but could be avoided if using MGCAMB/EFTCAMB/Hi_CLASS to get P(k) in MG instead
        k_camb, _, pk_camb = compute_pk.compute_pk_gr(params,[0.])
        D0_lambda = scipy.integrate.quadrature(growth_eqns.growth_int,1e-10,1.,args=(params['Omega_m'],1e-10,1.),tol=1e-8,rtol=1e-8)[0]
        Dz_lambda_norm = np.array([scipy.integrate.quadrature(growth_eqns.growth_int,1e-10,1/(1+zf),args=(params['Omega_m'],1e-10,1/(1+zf)),tol=1e-8,rtol=1e-8)[0] for zf in self.z])/D0_lambda
        Dz_reshaped = Dz_lambda_norm.reshape(len(Dz_lambda_norm),1)
        pk_camb_z = np.repeat(Dz_reshaped**2, len(pk_camb[0]), axis=1) * np.repeat(pk_camb, Nz, axis=0)
        logpk = [scipy.interpolate.CubicSpline(np.log(k_camb),np.log(pk_camb_z[i])) for i in range(Nz)]

        def pk(k,z_ix):
            return np.exp(logpk[z_ix](np.log(k)))

        k_sample = np.logspace(-4,np.log10(20),num=512)
        sigma2 = self.compute_sigma2(k_sample,Nz,pk)

        return sigma2, k_camb, pk_camb


    def sigma2_fr(self, params, kvec, pk_gr):
        """
        Compute a list of length Nz of sigma_lin^2(R,z[i]) functions for f(R) gravity given an input cosmology 
        and GR linear power spectrum at z=0
        """

        Nz = len(self.z)

        pk_vec = compute_pk.compute_pk_fr_vectorized(params['Omega_m'], params['fR0'], params['n'], self.z, kvec, pk_gr)
        logpk = [scipy.interpolate.CubicSpline(np.log(kvec),np.log(pk_vec[i])) for i in range(Nz)]

        def pk(k,z_ix):
            return np.exp(logpk[z_ix](np.log(k)))

        k_sample = np.logspace(-4,np.log10(20),num=512)
        sigma2 = self.compute_sigma2(k_sample,Nz,pk)

        return sigma2


    def sigma2_dgp(self, params, kvec, pk_gr):
        """
        Compute a list of length Nz of sigma_lin^2(R,z[i]) functions for nDGP gravity given an input cosmology 
        and GR linear power spectrum at z=0
        """

        Nz = len(self.z)

        pk_vec = compute_pk.compute_pk_dgp_vectorized(params['Omega_m'], params['rcH0'], self.z, pk_gr)
        logpk = [scipy.interpolate.CubicSpline(np.log(kvec),np.log(pk_vec[i])) for i in range(Nz)]

        def pk(k,z_ix):
            return np.exp(logpk[z_ix](np.log(k)))

        k_sample = np.logspace(-4,np.log10(20),num=512)
        sigma2 = self.compute_sigma2(k_sample,Nz,pk)

        return sigma2


    def get_pdf(self, s2_mu, sigma2):
        """
        Return matter PDF for a given matrix [i,j] of non-linear log-density variances for smoothing radii R[i] and 
        redshifts z[j] as well as a list of sigma_lin^2(R,z[i]) functions.
        """

        z = self.z
        R = self.R
        tau = self.tau
        Nz = len(z)
        Nr = len(R)

        rho_vec = np.linspace(0.1,13.5,1000)

        def get_psi(R, s2_mu, sigma2):
        # Here R, s2_mu and sigma2 refer to particular elements in the corresponding input arrays
            psi_vec = (sigma2(R)/s2_mu) * (tau(rho_vec)**2/(2*sigma2(R*rho_vec**(1/3))))
            return scipy.interpolate.CubicSpline(rho_vec, psi_vec)

        psi_mat = [[get_psi(rth, s2_mu[r_ix,z_ix], sigma2[z_ix]) for z_ix in range(Nz)] for r_ix,rth in enumerate(R)]

        def get_pdf_nn(psi):
        # compute non-normalised PDF
            pdf_nn_vec = np.sqrt((psi(rho_vec,2) + psi(rho_vec,1)/rho_vec)/(2*np.pi))*np.exp(-psi(rho_vec))
            return scipy.interpolate.CubicSpline(rho_vec, pdf_nn_vec)

        pdf_nn_mat = [[get_pdf_nn(psi_mat[r_ix][z_ix]) for z_ix in range(Nz)] for r_ix in range(Nr)]

        def mean_igr(rho, pdf_nn):
        # non-normalised PDF mean integrand
            return rho * pdf_nn(rho)

        mean_mat = [[scipy.integrate.quad(mean_igr,0.1,10,args=(pdf_nn_mat[r_ix][z_ix]),epsabs=1e-4,epsrel=1e-4,limit=50)[0] for z_ix in range(Nz)] for r_ix in range(Nr)]
        norm_mat = [[scipy.integrate.quad(pdf_nn_mat[r_ix][z_ix],0.1,10,epsabs=1e-4,epsrel=1e-4,limit=50)[0] for z_ix in range(Nz)] for r_ix in range(Nr)]

        def pdf(pdf_nn,mean,norm):
            rho_max = 10.
            pdf_vec = pdf_nn(rho_vec[rho_vec<rho_max] * mean/norm) * mean/norm**2
            return scipy.interpolate.CubicSpline(rho_vec[rho_vec<rho_max], pdf_vec)

        pdf_mat = [[pdf(pdf_nn_mat[r_ix][z_ix],mean_mat[r_ix][z_ix],norm_mat[r_ix][z_ix]) for z_ix in range(Nz)] for r_ix in range(Nr)]

        return pdf_mat


    def compute_pdf(self, cosmo_params, models=['gr']):
        """
        Compute matter PDF for requested cosmo_params and models. The output is a dictionary with keys 'gr' and/or 'fr' and/or 'dgp'. 
        Each value is a matrix [i][j] where [i] runs over (increasing) smoothing radius and [j] over (increasing) redshifts.
        """

        # compute linear and non-linear variances
        sig2_gr, k_camb, pk_camb = self.sigma2_gr(cosmo_params)
        s2_mu_gr = self.get_s2_mu(sig2_gr)
        if 'fr' in models: 
            sig2_fr = self.sigma2_fr(cosmo_params, k_camb, pk_camb)
            s2_mu_fr = self.get_s2_mu(sig2_fr)
        if 'dgp' in models: 
            sig2_dgp = self.sigma2_dgp(cosmo_params, k_camb, pk_camb)
            s2_mu_dgp = self.get_s2_mu(sig2_dgp)

        # compute matter PDF
        pdf_dict = {}
        if 'gr' in models:
            pdf_gr_mat = self.get_pdf(s2_mu_gr, sig2_gr)
            pdf_dict['gr'] = pdf_gr_mat
        if 'fr' in models: 
            pdf_fr_mat = self.get_pdf(s2_mu_fr, sig2_fr)
            pdf_dict['fr'] = pdf_fr_mat
        if 'dgp' in models: 
            pdf_dgp_mat = self.get_pdf(s2_mu_dgp, sig2_dgp)
            pdf_dict['dgp'] = pdf_dgp_mat
        
        return pdf_dict

    ######################### Test functions below #########################

    