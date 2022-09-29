#im_utils

# IM tools - convert to import statement

import numpy as np
import matplotlib.pyplot as plt
from astropy.cosmology import Planck15 as cosmo
from scipy.stats import chi2, norm
import scipy.special as spec

from classy import Class
from scipy.optimize import curve_fit
from scipy.special import eval_legendre

#n_modes tool

def get_nmodes(k,dk,V_s):
    """
    Returns number density of states for radial binning in k-space and survey volume V_s
    k: fourier wavenumber
    V_s: Survey volume
    dk: Size of k-space bin 
    """
    return ((k**2)*dk*V_s)/(4*(np.pi**2))

#binning in k-space for logspace k-grid

def get_dk(k_bin):
    """
    Returns dk, dlogk for uniform k_bin spacing 
    k_bin: k-binning, use np.logspace to generate unfirom k_binning
    """
    logk = np.log10(k_bin)
    dlogk = logk[1]-logk[0]
    dk = (10**(logk + (dlogk/2))) - (10**(logk - (dlogk/2)))
    return dk, dlogk

def get_expvals(k_bin, dk, k, P, bintype = 'linear'):
    """
    Returns the band averaged expectation values at bin centers. This is a weighted average by the number of modes
    across the band. 
    
    k_bin: centers of bin
    dk: linear bin widths
    k: Fourier wavenumber
    P: Power at fourier wavenumber k 
    """
    
    # need to compute mode weighted effective bin centers, initialize array to hold them
    # and initialize array of the expectation values 
    expvals = np.zeros(np.shape(k_bin)) #np.shape instead of np.len 
    k_eff = np.zeros(np.shape(k_bin))
    
    #loop over the k-bins and compute these things

    for ii in range(len(k_bin)):
        # Find bin edges
        if bintype == 'linear':
            lo = k_bin[ii] - (dk/2)
            hi = k_bin[ii] + (dk/2)
        if bintype == 'log':
            lo = 10**(np.log10(k_bin[ii]) - (dk/2))
            hi = 10**(np.log10(k_bin[ii]) + (dk/2))
        # Isolate input spectrum to these ks
        myinds = np.nonzero((k > lo) & (k < hi))
        myk = k[myinds[0]]
        myP = P[myinds[0]]
        
        # rescale by pretend 1st k in band has 1 mode, then scale by k^3
        nmodes = (myk/myk[0])**3
        # Take weighted average - also calc effective k
        expvals[ii] = np.sum(myP*nmodes) / np.sum(nmodes)
        k_eff[ii] = np.sum(myk*nmodes) / np.sum(nmodes)
    return expvals, k_eff

def inst_to_L(z, nu_rest, B, L):
    # Comoving radial distance R(z), Hubble parameter
    R = cosmo.comoving_distance(z)/cosmo.h
    H = cosmo.H(z)/cosmo.h
    # Parallel length
    L_par = (B * 2.998e5 * (1+z)**2)/(H.value * nu_rest)
    # Perp length
    L_perp = (L * R.value)/2
    return L_par, L_perp

#now need to explode errorbars at high k - from Kirits code

from scipy.integrate import quad

def get_resfac(z, nu_obs, sigma_b, delta_nu, k = None):
    # Comoving radial distance R(z), Hubble parameter 
    R = cosmo.comoving_distance(z)/cosmo.h
    H = cosmo.H(z)/cosmo.h
    # beam in arcminutes -> radians
    #beam = (theta_b/60.)*(np.pi/180.)
    sigma_perp = (R.value)*sigma_b#(R.value)*beam
    sigma_par = (2.998e5 / H.value) * (delta_nu*(1+z)/nu_obs)
    if k is None:
        resfac = None
    else:
        # Mu integral
        def integrand(mu, k, sigma_perp, sigma_par):
            return np.exp((mu**2)*(k**2)*((sigma_par**2)-(sigma_perp**2)))
        I = quad(integrand, 0, 1, args=(k,sigma_perp,sigma_par))
        resfac = np.exp((k*sigma_perp)**2) * I[0]
    return sigma_perp, sigma_par, resfac

# def get_resfac_RSD_multipoles(z, f, multipole, nu_obs, sigma_b, delta_nu, k = None):
#     # Comoving radial distance R(z), Hubble parameter 
#     R = cosmo.comoving_distance(z)/cosmo.h
#     H = cosmo.H(z)/cosmo.h
#     # beam in arcminutes -> radians
#     #beam = (theta_b/60.)*(np.pi/180.)
#     sigma_perp = (R.value)*sigma_b#(R.value)*beam
#     sigma_par = (2.998e5 / H.value) * (delta_nu*(1+z)/nu_obs)
#     if k is None:
#         resfac = None
#     else:
#         # Mu integral
#         def integrand(mu, k, sigma_perp, sigma_par):
#             return eval_legendre(multipole, mu)*(1 + f*mu**2)**2*np.exp((mu**2)*(k**2)*((sigma_par**2)-(sigma_perp**2)))
#         I = quad(integrand, 0, 1, args=(k,sigma_perp,sigma_par))
#         resfac = np.exp((k*sigma_perp)**2) * I[0]
#     return sigma_perp, sigma_par, resfac

def L_to_inst(z, nu_rest, L_par, L_perp):
    # Comoving radial distance R(z), Hubble parameter
    R = cosmo.comoving_distance(z)/cosmo.h
    H = cosmo.H(z)/cosmo.h
    # Bandwidth (see G12 above Eq. A6)
    B = (L_par * H.value * nu_rest)/(2.998e5 * (1+z)**2)
    # Matches Lidz 2016!
    # Survey length
    L = 2*L_perp / R.value #   I need a factor of 2 to match Lidz16
    return B, L

def extents(f):
    delta = f[1] - f[0]
    return [f[0] - delta/2, f[-1] + delta/2]

def detector_time(sigma_onedet, P_n, V_s, z_in, nu_rest, B, delta_nu):
    #sigma_onedet = 900.0
    l_par, l_perp = inst_to_L(z_in, nu_rest, delta_nu, theta_b)
    V_v = l_par * (l_perp**2) 
    sigma_pix = np.sqrt(P_n / V_v) 
    return (sigma_onedet/sigma_pix)**2 * (V_s/V_v) / (np.pi*1e7) / (B/delta_nu) #detector yrs for full volume



def P_n(t_s, sigma_pix = 600, theta = (1.0 / 60) * (np.pi / 180), N_s = 1000, A_s = 4*2500. * (np.pi / 180)**2, L_par=11.65, L_perp=0.44):
    V_vox = (L_perp**2) * L_par
    t_pix = t_s * (theta**2 / A_s)
    Pn = V_vox * (sigma_pix)**2 / (t_pix * N_s) 
    return Pn

def det_time(P_n, sigma_pix = 600, theta = (1.0 / 60) * (np.pi / 180), N_s = 1000, A_s = 4*2500. * (np.pi / 180)**2, L_par=11.65, L_perp=0.44):
    t_pix = (sigma_pix)**2/(P_n) 
    t_s = (t_pix*A_s)/(theta**2)
    return t_s

def A_perp(z_t, z_i):
    num = cosmo.angular_diameter_distance(z_t)
    denom = cosmo.angular_diameter_distance(z_i)
    return num/denom

def A_para(z_t, z_i):
    num = cosmo.H(z_t)*(1+z_t)
    denom = cosmo.H(z_i)*(1+z_i)
    return num/denom 

def prefactor(z_t, z_i):
    return ((A_perp(z_t, z_i))**2)*A_para(z_t, z_i)

def a(redshift):
    return cosmo.scale_factor(redshift)

def z(scale):
    return 1/scale - 1

def central_difference(f1, f2, h):
    return (f1 - f2)/(2*h)

def is_pos_def(A):
    if np.array_equal(A, A.T):
        try:
            np.all(np.linalg.eigvals(x) > 0)
            return True
        except: 
            print('not positive definite')
            return False
    else:
        print('not symmetric')
        return False