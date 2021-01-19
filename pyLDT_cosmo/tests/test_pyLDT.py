import numpy as np
import pyLDT_cosmo.pyLDT as pyLDT
from pkg_resources import resource_stream

def test_pdf():
    #load benchmarks
    rho = np.loadtxt(resource_stream('pyLDT_cosmo', 'benchmarks/rho.dat'))
    pdf_z0 = np.loadtxt(resource_stream('pyLDT_cosmo', 'benchmarks/pdf_gr_z0.0_R10.dat'))
    pdf_z0p5 = np.loadtxt(resource_stream('pyLDT_cosmo', 'benchmarks/pdf_gr_z0.5_R10.dat'))
    pdf_z1 = np.loadtxt(resource_stream('pyLDT_cosmo', 'benchmarks/pdf_gr_z1.0_R10.dat'))

    z = np.array([0., 0.5, 1.]) # output redshifts
    R = np.array([10.]) # radius of top-hat smoothing filter in Mpc/h
    cosmo_params_fid = {'Omega_m': 0.31315, 'Omega_b': 0.0492, 'A_s': 2.0968e-9, 'n_s': 0.9652, 'h': 67.37}
    s2_mu_gr_fid = np.array([[0.391930, 0.252758, 0.166748]])
    sig2_gr_fid = pyLDT.init_fid(cosmo_params_fid, z)

    # Set cosmological parameters
    cosmo_params = cosmo_params_fid
    # Compute matter PDF for requested models
    pdf_mat = pyLDT.compute_pdf(cosmo_params, z, R, sig2_gr_fid, s2_mu_gr_fid)

    assert np.allclose(pdf_mat['gr'][0][0](rho), pdf_z0)
    assert np.allclose(pdf_mat['gr'][0][1](rho), pdf_z0p5)
    assert np.allclose(pdf_mat['gr'][0][2](rho), pdf_z1)

if __name__ == "__main__":
    test_pdf()
