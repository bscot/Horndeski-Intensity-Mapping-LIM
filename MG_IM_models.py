import numpy as np
import matplotlib.pyplot as plt
from astropy.cosmology import Planck15 as cosmo
from scipy.stats import chi2, norm
import scipy.special as spec

from classy import Class
from scipy.optimize import curve_fit

from bryan_im_utils import *

class Survey:
    '''
    The Survey object contains the experiment/survey definition

    Args:
        Bandwidth (str): The arg is used for...
        Dish_size (str):
        theta_b ():
        field_length_deg (): 
        freq_bin_width (): 
        k_List (): 
        
        *args: The variable arguments are used for...
        **kwargs: The keyword arguments are used for...
    
    Methods: 
        set_targets:
        set_survey_dimensions:
        set_survey_Volume: 
        nmodes: 
    
    Attributes:
        B, D, FWHM, sigma_b, field_length, delta_nu, k_List (floats): Survey parameters, see __init__ docstring for detailed information. 
        z, nu_rest, k_bin, I, bList, zList (floats): Target parameters, see set_targets for detailed information. 
    '''

    def __init__(self, Bandwidth, Dish_size, theta_b, field_length_deg, freq_bin_width, k_List):
        
        '''
        Initializes the Survey object with basic survey parameter definitions.
        
        Attributes: 
        B (float): 
        D (float):
        FWHM (float):
        sigma_b (float):
        field_length (float):
        delta_nu (float): 
        kList (float): 
        
        '''
        
        self.B = Bandwidth
        self.D = Dish_size
        self.FWHM = theta_b
        self.sigma_b = theta_b/2.35
        self.field_length = field_length_deg
        self.delta_nu = freq_bin_width
        self.kList = k_List
        #self.kList = ... 
       
        
    def set_targets(self, redshift, target_line_rest_freq, line_intensity, k_min, k_max,bins, b_list, z_list):
        
        '''
        
        Sets targets for the survey. 
        
        Attributes:
        z (float): stores target line redshift
        nu_rest (float): stores target line restframe frequency
        k_bin (numpy array): stores Power spectra binning, assumes logarithmic bins 
        I (float): stores target line intensity, units match/set white noise level units in noise model.
        bList (array/list): target line biases #check this
        zList (array/list): target line redshifts #check this
        
        '''
        
        self.z = redshift
        self.nu_rest = target_line_rest_freq 
        self.k_bin = np.logspace(k_min, k_max, bins)
        self.I = line_intensity
        self.bList = b_list
        self.zList = z_list
        
    def set_survey_dimensions(self):
        
        '''Defines the 'box size' or survey dimensions, L_par, L_perp in physical units (Mpc) based on experimental parameters'''
   
        L_par, L_perp = inst_to_L(self.z, self.nu_rest, self.B, self.field_length*(np.pi/180))
        return L_par, L_perp
        
    def set_survey_Volume(self):
        
        '''
        Computes the survey volume from the box size, L_par, L_perp in physical units (Mpc) based on experimental parameters. Used by nmodes/cosmic variance calculation.
        '''
        
        L_par2, L_perp2 = self.set_survey_dimensions()
        self.V_s = L_par2 * L_perp2**2
        return self.V_s
    
    def nmodes(self):
        '''
        Computes the number of modes in the box, nmodes based on the survey volume, used for the cosmic variance calculation.
        '''
        
        self.dk, self.dlogk = get_dk(self.k_bin) 
        self.V_s = self.set_survey_Volume()
        nmodes = get_nmodes(self.k_bin,self.dk,self.V_s) 
        return 2*nmodes #changed to 2*Nmodes for dual polarization instrument
    
class Modified_Gravity_Model():
    '''
    The Modified_Gravity_Model object contains the modified gravity model definition. 

    Args:
        x_k (float): Kineticity Evolution - only effects scales ~ horizon, fixed to unity
        x_b (float): Braiding Evolution 
        x_m (float): Planck/Gravitational mass run rate
        x_t (float): Tensor speed excess (1+x_t) - fixed to zero
        M (): Planck Mass (natural units)
        k_List (): 
        
        *args: The variable arguments are used for...
        **kwargs: The keyword arguments are used for...
    
    Methods: 
        set_model: updates fiducial values of the x_X functions. 
        compute_MG_Pk: computes the Power Spectra for a given value of the x_X functions 
        compute fsigma8: computes fsigma8 for the RSD constraints on the MG functions 
    
    Attributes:
        x_k, x_b, x_m, x_t, M (floats): Modified gravity rate functions
        Pk (array): Matter Power Spectra in a given MG model at fixed redshift
        fsigma8 (float): value of fsigma8 in a given MG model at fixed redshift 

    '''
    
    def __init__(self):
        
        '''
        Initializes the Modified_Gravity_Model object with sensible model parameter values. 
        '''
        
        # set these to something sensible! 
        
        self.x_k = 1
        self.x_b = 0.63 #have been using 0.6, should make sure all are rerun to 0.63 standard 
        self.x_m = 0.20
        self.x_t = 0
        self.M = 1
        
    def set_model(self, model, x_k, x_b, x_m, x_t, M):
        
        '''
        Updates the Modified_Gravity_Model object with new model parameter values. 
        '''
        
        self.model = model
        self.x_k = x_k
        self.x_b = x_b 
        self.x_m = x_m
        self.x_t = x_t
        self.M = M
        
    def compute_MG_Pk(self, redshift, kList):
        
        '''
        Computes the matter Power Spectra at a given redshift and over a given set of values in k-space. Updates Pk attribute of the Modified gravity model with the values of the matter power spectra. 
        
        Args:
        redshift (float): redshift at which to compute the matter power spectra
        kList (array/list): set of values to compute the matter power spectra at in k-space.
        
        Returns:
        Pk (array): the values of the matter power spectra at given values in k-space by kList. 
        
        '''
                
        modified_gravity_params = {
            'Omega_Lambda': 0,
            'Omega_fld': 0,
            'Omega_smg': -1,
            'expansion_model': 'lcdm',
            'output': 'tCl,pCl,lCl,mPk', 
            'P_k_max_1/Mpc':3.0, 
            'z_max_pk': 9.0
            }
        
        if self.model == 'a':
#             print('a')
            modified_gravity_params['gravity_model'] = 'propto_scale'
            
        if self.model == 'DE':
            print('DE')
            modified_gravity_params['gravity_model'] = 'propto_omega'
        
    
        mg_cosmo = Class()
        modified_gravity_params['parameters_smg'] = "{}, {}, {}, {}, {}".format(self.x_k, self.x_b, self.x_m, self.x_t, self.M)
        self.modified_gravity_params = modified_gravity_params
#         print(modified_gravity_params)
        mg_cosmo.set(modified_gravity_params)
        mg_cosmo.compute()
    
        MG_Pk_best = np.array([mg_cosmo.pk(k,redshift) for k in kList])
        
        self.Pk = MG_Pk_best
        return self.Pk
    
    def compute_fsigma8(self, redshift):
        
        '''
        Computes the value of fsigma8 at a given redshift. Updates fsigma8 attribute of the Modified gravity model. 
        Used in the RSD constraints. 
        
        Args:
        redshift (float): redshift at which to compute fsigma8. 
        
        Returns:
        fsigma (float): the value of fsigma8 at a given redshift.
        
        '''  
        
        
        modified_gravity_params = {
            'Omega_Lambda': 0,
            'Omega_fld': 0,
            'Omega_smg': -1,
            'expansion_model': 'lcdm',
            'output': 'tCl,pCl,lCl,mPk', 
            'P_k_max_1/Mpc':3.0, 
            'z_max_pk': 9.0
        }
        
        if self.model == 'a':
            print('a')
            modified_gravity_params['gravity_model'] = 'propto_scale'
            
        if self.model == 'DE':
            print('DE')
            modified_gravity_params['gravity_model'] = 'propto_omega'
        
    
        mg_cosmo = Class()
        modified_gravity_params['parameters_smg'] = "{}, {}, {}, {}, {}".format(self.x_k, self.x_b, self.x_m, self.x_t, self.M)
        self.modified_gravity_params = modified_gravity_params
#         print(modified_gravity_params)
        mg_cosmo.set(modified_gravity_params)
        mg_cosmo.compute()
        
        #scale independent growth rate: 
        # print('Survey.z is: ')
        # print(Survey.z)
        mg_fsigma8 = np.array(mg_cosmo.scale_independent_growth_factor_f(redshift)*mg_cosmo.sigma(8, redshift))
        print('mg_fsigma8 is: ')
        print(mg_fsigma8)
        
        # note, this may not play well with the f_RSD function, adapt as necessary 
        self.f = mg_cosmo.scale_independent_growth_factor_f(redshift)
        
        #scale depedendent growth rate
        
        self.sigma8 = mg_cosmo.sigma(8, redshift) 
        
        return mg_fsigma8
    
class Noise_Model:
    
    """
    Defines and updates the noise model for use in the forecast.
    
    Args:
        Pn (float): White (gaussian) Noise level
        alpha (float):  Atmospheric noise exponential parameter (default 5/3)
        Beta (float): Continuum noise exponential smoothing parameter (default 5/3)
        l_knee (float): Atmospheric angular power specta cutoff, converted internally to k-space
        eta_min (float): Continuum noise cutoff, converted internally to k-space
        I (float): Target line intensity, units match/set those of Pn. 
        
        *args: The variable arguments are used for...
        **kwargs: The keyword arguments are used for...
    
    Methods: 
        set_Noise_model: sets the Noise model. See function docstring for options. 
        interloper: computes interloper power spectra. See function docstring for args/parameters. 
        sigma_ : (model explanation goes here). 
        
    Attributes:
        Pn, alpha, Beta, l_knee, eta_min, I (floats): Noise model parameters
        sigma (array): value of noise (gaussian width parameter) at each k_bin center of binned power spectra. 
    
    """
    
    def __init__(self, Pn, P_shot, alpha, l_knee, Beta, eta_min, line_intensity):
        
        '''
        Initializes noise model with values for the noise parameters. 
        '''
        
        
        self.Pn = Pn
        self.alpha = alpha
        self.Beta = Beta
        self.l_knee = l_knee
        self.eta_min = eta_min
        self.I = line_intensity
        
        self.Pn = Pn
        self.shot = P_shot
        
    def set_Noise_model(self, Noise_model, Survey, Modified_Gravity_Model):
        
        '''
        Sets noise model and updates sigma attribute with the value of the noise (gaussian width parameter 'sigma') at the bin centers of the binned matter power spectra or RSD power spectra monopole/quadrupole. 
        
        '''
        
    
        elif Noise_model == 'RSD_mono':
            self.sigma = self.sigma_RSD_mono(Survey, Modified_Gravity_Model)
            print(Noise_model)
        elif Noise_model == 'RSD_quad':
            self.sigma = self.sigma_RSD_quad(Survey, Modified_Gravity_Model)
            print(Noise_model)
        elif Noise_model == 'RSD_mono_interloper':
            self.sigma = self.sigma_RSD_mono_interloper_unmitigated(Survey, Modified_Gravity_Model)
            print(Noise_model)
        elif Noise_model == 'RSD_quad_interloper':
            self.sigma = self.sigma_RSD_quad_interloper_unmitigated(Survey, Modified_Gravity_Model)
            print(Noise_model)
        elif Noise_model == 'fiducial_mitigated':
            self.sigma = self.sigma_fiducial_mitigated(Survey, Modified_Gravity_Model)
            print(Noise_model)
        elif Noise_model == 'fiducial_unmitigated':
            self.sigma = self.sigma_fiducial_unmitigated(Survey, Modified_Gravity_Model)
            print(Noise_model)    
        elif Noise_model == 'covariance':
            self.sigma = self.sigma_RSD_mono_quad_cov(Survey, Modified_Gravity_Model) 
            print(Noise_model) 
        elif Noise_model == 'covariance_full': 
            self.sigma = self.combined_cov(Survey, Modified_Gravity_Model) 
            print(Noise_model)
        else:
            print('Not a noise model.')
            
        print(self.sigma)
    
    def interloper(self, Survey, Modified_Gravity_Model, bias, redshift):
        """
        Computes 'interloper' power spectra * bI^2. Returns an array containing the matter power spectra multiplied by the interloper line intensity at the interloper redshift. 
        
        Parameters:
        Survey (Survey object): Survey/experiment object holding survey definition.
        Modified_Gravity_Model (Modified Gravity Model object): Modified Gravity model object holding the model definition
        bias (float): actually the line intensity. 
        redshift (float): interloper line redshift. 
        """
        
        x_k = Modified_Gravity_Model.x_k
        x_b = Modified_Gravity_Model.x_b
        x_m = Modified_Gravity_Model.x_m
        x_t = Modified_Gravity_Model.x_t
        M = Modified_Gravity_Model.M
    
        interloper_cosmo = Class()
        modified_gravity_params = Modified_Gravity_Model.modified_gravity_params
        modified_gravity_params['parameters_smg'] = "{}, {}, {}, {}, {}".format(x_k, x_b, x_m, x_t, M)
        interloper_cosmo.set(modified_gravity_params)
        interloper_cosmo.compute()
    
        MG_Pk_best = np.array([interloper_cosmo.pk(k,redshift) for k in Survey.kList])

    #     MG_expvals, k_eff = get_expvals(k_bin, dlogk, kList, MG_Pk_best, bintype = 'log')
    
        bI = (1+redshift)*bias
        bI2 = bI*bI
    
        return bI2*MG_Pk_best
    
    
###################################################### 1. mPk w/Interlopers ######################################################################


    def sigma_fiducial_unmitigated(self, Survey, Modified_Gravity_Model):
        
        """   
        Computes 'all' noise model *with* geometric correction for interlopers. Returns an array containing the noise at the bin centers. 
        Included Noise components:
        1. Pn "white"
        2. Finite Instrument Spectral Resolution
        
        Edited to exclude: 
        3. Interlopers w/ geometric correction
        
        Parameters:
        Survey (Survey object): Survey/experiment object holding survey definition.
        Modified_Gravity_Model (Modified Gravity Model object): Modified Gravity model object holding the model definition
        
        """
        
        b = (1+Survey.z)
        bI = (1+Survey.z)*self.I
        bI2 = bI*bI
        
        nmodes = Survey.nmodes()
        
        
        expvals, k_eff = get_expvals(Survey.k_bin, Survey.dlogk, Survey.kList, Modified_Gravity_Model.Pk, bintype = 'log')
        self.k_eff = k_eff
    
        # white
    
        P_n_bin = self.Pn * np.ones(np.shape(k_eff)) 
    
        # + smoothing 
    
        for kk in range(len(k_eff)):
            sigma_perp, sigma_par, resfac = get_resfac(Survey.z, Survey.nu_rest/(1+Survey.z), Survey.sigma_b, Survey.delta_nu,
                                                       k = k_eff[kk])
            P_n_bin[kk] = P_n_bin[kk]*resfac 
        
        # + 1/f
        
        for kk,k in enumerate(k_eff):
            kstol = k*cosmo.comoving_distance(0.5)/cosmo.h
            P_n_bin[kk] = P_n_bin[kk]*( 1 + (np.array(kstol)/self.l_knee)**(self.alpha))
            
        # shot
        P_n_bin = P_n_bin + self.shot * np.ones(np.shape(k_eff)) 
        
        # # + foregrounds
        # for kk, k in enumerate(k_eff):
        #     P_n_bin[kk] = P_n_bin[kk] + expvals[kk]*(1 + (k/self.eta_min)**self.Beta)
        
        ######### mitigated interlopers #########
    
#         interlopers = np.zeros(len(Survey.kList))
        
#         for i in range(0, len(Survey.bList)):
#             interlopers = interlopers + prefactor(Survey.z, Survey.zList[i])*self.interloper(Survey, Modified_Gravity_Model, Survey.bList[i], Survey.zList[i])
        
#         interlopers_Pk, k_eff = get_expvals(Survey.k_bin, Survey.dlogk, Survey.kList, interlopers, bintype = 'log')
        
        ######### unmitigated interlopers #########
    
        interlopers = np.zeros(len(Survey.kList))
        
        for i in range(0, len(Survey.bList)):
            interlopers = interlopers + self.interloper(Survey, Modified_Gravity_Model, Survey.bList[i], Survey.zList[i])
        
        interlopers_Pk, k_eff = get_expvals(Survey.k_bin, Survey.dlogk, Survey.kList, interlopers, bintype = 'log')
        
        #sigma = (bI21*expvals + P_n_bin)/np.sqrt(nmodes)
    
        sigma = (bI2*expvals + P_n_bin + interlopers_Pk)/np.sqrt(nmodes)
        # sigma = (bI2*expvals + P_n_bin)/np.sqrt(nmodes)
        print(sigma)

        return sigma
    
###################################################### 2. mPk No Interlopers ######################################################################


    
    def sigma_fiducial_mitigated(self, Survey, Modified_Gravity_Model):
        
        """   
        Computes 'all' noise model *with* geometric correction for interlopers. Returns an array containing the noise at the bin centers. 
        Included Noise components:
        1. Pn "white"
        2. Finite Instrument Spectral Resolution
        
        Edited to exclude: 
        3. Interlopers w/ geometric correction
        
        Parameters:
        Survey (Survey object): Survey/experiment object holding survey definition.
        Modified_Gravity_Model (Modified Gravity Model object): Modified Gravity model object holding the model definition
        
        """
        
        b = (1+Survey.z)
        bI = (1+Survey.z)*self.I
        bI2 = bI*bI
        
        nmodes = Survey.nmodes()
        
        
        expvals, k_eff = get_expvals(Survey.k_bin, Survey.dlogk, Survey.kList, Modified_Gravity_Model.Pk, bintype = 'log')
        self.k_eff = k_eff
    
        # white
    
        P_n_bin = self.Pn * np.ones(np.shape(k_eff)) 
        
        # + 1/f 
    
#         for kk,k in enumerate(k_eff):
#             kstol = k*cosmo.comoving_distance(0.5)/cosmo.h
#             P_n_bin[kk] = P_n_bin[kk]*( 1 + (np.array(kstol)/self.l_knee)**(self.alpha))
        
#         # + foregrounds
#         for kk, k in enumerate(k_eff):
#             P_n_bin[kk] = P_n_bin[kk] * (1 + (k/self.eta_min)**self.Beta)
    
        # + smoothing 
    
        for kk in range(len(k_eff)):
            sigma_perp, sigma_par, resfac = get_resfac(Survey.z, Survey.nu_rest/(1+Survey.z), Survey.sigma_b, Survey.delta_nu,
                                                       k = k_eff[kk])
            P_n_bin[kk] = P_n_bin[kk]*resfac

            
        P_n_bin = P_n_bin + self.shot * np.ones(np.shape(k_eff)) 
        
        ######### mitigated interlopers #########
    
#         interlopers = np.zeros(len(Survey.kList))
        
#         for i in range(0, len(Survey.bList)):
#             interlopers = interlopers + AP_prefactor(Survey.z, Survey.zList[i])*self.interloper(Survey, Modified_Gravity_Model, Survey.bList[i], Survey.zList[i])
        
#         interlopers_Pk, k_eff = get_expvals(Survey.k_bin, Survey.dlogk, Survey.kList, interlopers, bintype = 'log')
        
        ######### unmitigated interlopers #########
    
#         interlopers = np.zeros(len(Survey.kList))
        
#         for i in range(0, len(Survey.bList)):
#             interlopers = interlopers + self.interloper(Survey, Modified_Gravity_Model, Survey.bList[i], Survey.zList[i])
        
#         interlopers_Pk, k_eff = get_expvals(Survey.k_bin, Survey.dlogk, Survey.kList, interlopers, bintype = 'log')
        
        sigma = (bI2*expvals + P_n_bin)/np.sqrt(nmodes)
    
        # sigma = (bI2*expvals + P_n_bin + interlopers_Pk)/np.sqrt(nmodes)
        # sigma = (bI2*expvals + P_n_bin)/np.sqrt(nmodes)
        print(sigma)

        return sigma
    
###################################################### 3. Monopole No Interlopers ######################################################################


    
    def sigma_RSD_mono(self, Survey, Modified_Gravity_Model):
        
        """
        Computes the Redshift Space Distortion (RSD) monopole errors with no interlopers included. Returns an array containing the noise at the bin centers. 
        Included Noise components:
        1. Pn "white"
        2. Finite Instrument Spectral Resolution
        
        Parameters:
        Survey (Survey object): Survey/experiment object holding survey definition.
        Modified_Gravity_Model (Modified Gravity Model object): Modified Gravity model object holding the model definition
        
        """
        
        b = (1+Survey.z)
        bI = (1+Survey.z)*self.I
        bI2 = bI*bI
        
        nmodes = Survey.nmodes()
        print(Survey.delta_nu)
#         f_int = Modified_Gravity_Model.f
#         sigma_int = Modified_Gravity_Model.sigma8
#         sigma_int2 = sigma_int**2
        
#         print('Fiducial values: ')
#         print(f_int)
#         print(sigma_int)
        
        fsigma8 = Modified_Gravity_Model.compute_fsigma8(Survey.z)
        Beta = Modified_Gravity_Model.f/b
    
        expvals, k_eff = get_expvals(Survey.k_bin, Survey.dlogk, Survey.kList, Modified_Gravity_Model.Pk, bintype = 'log')
        self.k_eff = k_eff
            
        # The uncertainty on the monopole and quadrupole are sigma_mono = Pl0 + Pn/np.sqrt(2*nmodes) and sigma_quad = Pl2 + Pn/np.sqrt(2*nmodes)
        print(self.Pn)
        P_n_bin = self.Pn * np.ones(np.shape(k_eff)) 
        
        for kk in range(len(k_eff)):
            sigma_perp, sigma_par, resfac = get_resfac_RSD_multipoles(Survey.z, fsigma8,0,Survey.nu_rest/(1+Survey.z), Survey.sigma_b, Survey.delta_nu, k = k_eff[kk])
#             P_k_bin_mono[kk] = expvals_mono[kk]*resfac
            P_n_bin[kk] = P_n_bin[kk]*resfac
        
        print(P_n_bin)
        #shot 
                                    
        P_n_bin = P_n_bin + self.shot * np.ones(np.shape(k_eff)) 

        
        pre_factor = 2
        
        # check factor of f vs beta here
        
        first_term = (1 + 4/3*Beta + 6/5*Beta**2 + 4/7*Beta**3 + 1/9*Beta**4)*(bI2*expvals)**2
        second_term = 2*P_n_bin*(1+2/3*Beta + 1/5 * Beta**2)*bI2*expvals + P_n_bin**2
        
        numerator = pre_factor*(first_term + second_term)
    
        sigma_RSD_mono = np.sqrt(numerator)/(np.sqrt(nmodes))
        
        return sigma_RSD_mono
    
###################################################### 4. Quadrupole No Interlopers ######################################################################


    
    def sigma_RSD_quad(self, Survey, Modified_Gravity_Model):
        
        """
        Computes the Redshift Space Distortion (RSD) quadrupole errors with no interlopers included. Returns an array containing the noise at the bin centers. 
        Included Noise components:
        1. Pn "white"
        2. Finite Instrument Spectral Resolution
        
        Parameters:
        Survey (Survey object): Survey/experiment object holding survey definition.
        Modified_Gravity_Model (Modified Gravity Model object): Modified Gravity model object holding the model definition
        
        """
        
        b = (1+Survey.z)
        bI = (1+Survey.z)*self.I
        bI2 = bI*bI
        
        nmodes = Survey.nmodes()
        
#         f_int = Modified_Gravity_Model.f
#         sigma_int = Modified_Gravity_Model.sigma8
#         sigma_int2 = sigma_int**2
        
#         print('Fiducial values: ')
#         print(f_int)
#         print(sigma_int)
        
        fsigma8 = Modified_Gravity_Model.compute_fsigma8(Survey.z)
        Beta = Modified_Gravity_Model.f/b
        
        expvals, k_eff = get_expvals(Survey.k_bin, Survey.dlogk, Survey.kList, Modified_Gravity_Model.Pk, bintype = 'log')
        self.k_eff = k_eff
            
        # The uncertainty on the monopole and quadrupole are sigma_mono = Pl0 + Pn/np.sqrt(2*nmodes) and sigma_quad = Pl2 + Pn/np.sqrt(2*nmodes)
        print(self.Pn)
        
        P_n_bin = self.Pn * np.ones(np.shape(k_eff))
        
        for kk in range(len(k_eff)):
            sigma_perp, sigma_par, resfac = get_resfac(Survey.z, Survey.nu_rest/(1+Survey.z), Survey.sigma_b, Survey.delta_nu, k = k_eff[kk])
#             P_k_bin_mono[kk] = expvals_mono[kk]*resfac
            P_n_bin[kk] = P_n_bin[kk]*resfac
       
        #shot
                                    
        P_n_bin = P_n_bin + self.shot * np.ones(np.shape(k_eff)) 
        
        prefactor = 2
        first_term = (5 + 220/21 * Beta + 90/7 * Beta**2 + 1700/231 * Beta**3 + 2075/1287 * Beta**4)*(bI2*expvals)**2
        second_term = 2*P_n_bin*(5 + 220/21*Beta + 30/7 * Beta**2)*bI2*expvals
        third_term = 5*P_n_bin**2 
    
        numerator = prefactor*(first_term + second_term + third_term) 
        
        sigma_RSD_quad = np.sqrt(numerator)/(np.sqrt(nmodes))
        
        return sigma_RSD_quad 
    
###################################################### 5. monopole w/ Interlopers ######################################################################

    
    def sigma_RSD_mono_interloper_unmitigated(self, Survey, Modified_Gravity_Model):
        
        """
        Computes the Redshift Space Distortion (RSD) monopole errors *with* interlopers included. Returns an array containing the noise at the bin centers. 
        Included Noise components:
        1. Pn "white"
        2. Finite Instrument Spectral Resolution
        3. interlopers
        
        Parameters:
        Survey (Survey object): Survey/experiment object holding survey definition.
        Modified_Gravity_Model (Modified Gravity Model object): Modified Gravity model object holding the model definition
        
        """
        
        b = (1+Survey.z)
        bI = (1+Survey.z)*self.I
        bI2 = bI*bI
               
        nmodes = Survey.nmodes()
        
#         f_int = Modified_Gravity_Model.f
#         sigma_int = Modified_Gravity_Model.sigma8
#         sigma_int2 = sigma_int**2
        
#         print('Fiducial values: ')
#         print(f_int)
#         print(sigma_int)
        
        fsigma8 = Modified_Gravity_Model.compute_fsigma8(Survey.z)
        Beta = Modified_Gravity_Model.f/b
    
        expvals, k_eff = get_expvals(Survey.k_bin, Survey.dlogk, Survey.kList, Modified_Gravity_Model.Pk, bintype = 'log')
        self.k_eff = k_eff
            
        # The uncertainty on the monopole and quadrupole are sigma_mono = Pl0 + Pn/np.sqrt(2*nmodes) and sigma_quad = Pl2 + Pn/np.sqrt(2*nmodes)
        print(self.Pn)
        P_n_bin = self.Pn  * np.ones(np.shape(k_eff)) 
        
        expvals_mono = (1 + 2/3*fsigma8 + 1/5*(fsigma8)**2)*bI2*expvals #update this line
        P_k_bin_mono = np.ones(len(expvals_mono))
        
        for kk in range(len(k_eff)):
            sigma_perp, sigma_par, resfac = get_resfac(Survey.z, Survey.nu_rest/(1+Survey.z), Survey.sigma_b, Survey.delta_nu, k = k_eff[kk])
#             P_k_bin_mono[kk] = expvals_mono[kk]*resfac
            P_n_bin[kk] = P_n_bin[kk]*resfac
        
        P_n_bin = P_n_bin + self.shot * np.ones(np.shape(k_eff)) 

        interlopers_var = np.zeros(len(k_eff))
        
        for i in range(0, len(Survey.bList)):
            fsigma8 = Modified_Gravity_Model.compute_fsigma8(Survey.zList[i])
            Beta_int = Modified_Gravity_Model.f/b
            interlopers = self.interloper(Survey, Modified_Gravity_Model, Survey.bList[i], Survey.zList[i])
            
            interlopers_Pk, k_eff = get_expvals(Survey.k_bin, Survey.dlogk, Survey.kList, interlopers, bintype = 'log')
            
            pre_factor = 2
            first_term = (1 + 4/3*Beta_int + 6/5*Beta_int**2 + 4/7*Beta_int**3 + 1/9*Beta_int**4)*(interlopers_Pk)**2
            second_term = 2*P_n_bin*(1+2/3*Beta_int + 1/5 * Beta_int**2)*(interlopers_Pk) + P_n_bin**2
            interlopers_var = interlopers_var + pre_factor*(first_term + second_term) 

        
        pre_factor = 2
        first_term = (1 + 4/3*Beta + 6/5*Beta**2 + 4/7*Beta**3 + 1/9*Beta**4)*(bI2*expvals)**2
        second_term = 2*P_n_bin*(1+2/3*Beta + 1/5 * Beta**2)*(bI2*expvals) + P_n_bin**2
        
        numerator = (first_term + second_term) + interlopers_var
        
        sigma_RSD_mono_interlopers = np.sqrt(numerator)/(np.sqrt(nmodes))
        
        return sigma_RSD_mono_interlopers 
    
###################################################### 6. Quadrupole w/Interlopers ######################################################################
    
    def sigma_RSD_quad_interloper_unmitigated(self, Survey, Modified_Gravity_Model):
        
        """
        Computes the Redshift Space Distortion (RSD) quadrupole errors *with* interlopers included. Returns an array containing the noise at the bin centers. 
        Included Noise components:
        1. Pn "white"
        2. Finite Instrument Spectral Resolution
        3. interlopers
        
        Parameters:
        Survey (Survey object): Survey/experiment object holding survey definition.
        Modified_Gravity_Model (Modified Gravity Model object): Modified Gravity model object holding the model definition
        
        """
        
        b = (1+Survey.z)
        bI = (1+Survey.z)*self.I
        bI2 = bI*bI
        
        # Beta = Modified_Gravity_Model.f/b
        
        nmodes = Survey.nmodes()
        
        fsigma8 = Modified_Gravity_Model.compute_fsigma8(Survey.z)
        Beta = Modified_Gravity_Model.f/b
        
        expvals, k_eff = get_expvals(Survey.k_bin, Survey.dlogk, Survey.kList, Modified_Gravity_Model.Pk, bintype = 'log')
        self.k_eff = k_eff
            
        # The uncertainty on the monopole and quadrupole are sigma_mono = Pl0 + Pn/np.sqrt(2*nmodes) and sigma_quad = Pl2 + Pn/np.sqrt(2*nmodes)
        print(self.Pn)
        P_n_bin = self.Pn * np.ones(np.shape(k_eff)) 
        
        expvals_quad = (4/3*fsigma8 + 4/7*fsigma8**2)*bI2*expvals #update this line
        P_k_bin_quad = np.ones(len(expvals_quad))
        
        for kk in range(len(k_eff)):
            sigma_perp, sigma_par, resfac = get_resfac(Survey.z, Survey.nu_rest/(1+Survey.z), Survey.sigma_b, Survey.delta_nu, k = k_eff[kk])
#             P_k_bin_mono[kk] = expvals_mono[kk]*resfac
            P_n_bin[kk] = P_n_bin[kk]*resfac
        
        P_n_bin = P_n_bin + self.shot * np.ones(np.shape(k_eff)) 

        interlopers_var = np.zeros(len(k_eff))
        
        pre_factor = 2
        
        for i in range(0, len(Survey.bList)):
            fsigma8 = Modified_Gravity_Model.compute_fsigma8(Survey.zList[i])
            Beta_int = Modified_Gravity_Model.f/b
            interlopers = self.interloper(Survey, Modified_Gravity_Model, Survey.bList[i], Survey.zList[i])
            
            interlopers_Pk, k_eff = get_expvals(Survey.k_bin, Survey.dlogk, Survey.kList, interlopers, bintype = 'log')

            pre_factor = 2
            first_term = (5 + 220/21 * Beta_int + 90/7 * Beta_int**2 + 1700/231 * Beta_int**3 + 2075/1287 * Beta_int**4)*(interlopers_Pk)**2
            second_term = 2*P_n_bin*(5 + 220/21*Beta_int + 30/7 * Beta_int**2)*(interlopers_Pk)
            third_term = 5*P_n_bin**2 
            
            interlopers_var = interlopers_var + pre_factor*(first_term + second_term + third_term) 
        
    
        first_term = (5 + 220/21 * Beta + 90/7 * Beta**2 + 1700/231 * Beta**3 + 2075/1287 * Beta**4)*(bI2*expvals)**2
        second_term = 2*P_n_bin*(5 + 220/21*Beta + 30/7 * Beta**2)*(bI2*expvals)
        third_term = 5*P_n_bin**2 
        numerator_exp = first_term + second_term + third_term 
    
        numerator = numerator_exp + interlopers_var
        
        sigma_RSD_quad = np.sqrt(numerator)/(np.sqrt(nmodes))
        
        return sigma_RSD_quad     
    
###################################################### 7. RSD Covariance No Interlopers ######################################################################

    
    def sigma_RSD_mono_quad_cov(self, Survey, Modified_Gravity_Model):
        
        """
        Computes the Redshift Space Distortion (RSD) quadrupole errors *with* interlopers included. Returns an array containing the noise at the bin centers. 
        Included Noise components:
        1. Pn "white"
        2. Finite Instrument Spectral Resolution
        3. interlopers
        
        Parameters:
        Survey (Survey object): Survey/experiment object holding survey definition.
        Modified_Gravity_Model (Modified Gravity Model object): Modified Gravity model object holding the model definition
        
        """
        
        b = (1+Survey.z)
        bI = (1+Survey.z)*self.I
        bI2 = bI*bI
        
        nmodes = Survey.nmodes()
        
#         f_int = Modified_Gravity_Model.f
#         sigma_int = Modified_Gravity_Model.sigma8
#         sigma_int2 = sigma_int**2
        
#         print('Fiducial values: ')
#         print(f_int)
#         print(sigma_int)
        
        fsigma8 = Modified_Gravity_Model.compute_fsigma8(Survey.z)
        Beta = Modified_Gravity_Model.f/b
        
        expvals, k_eff = get_expvals(Survey.k_bin, Survey.dlogk, Survey.kList, Modified_Gravity_Model.Pk, bintype = 'log')
        self.k_eff = k_eff
            
        # The uncertainty on the monopole and quadrupole are sigma_mono = Pl0 + Pn/np.sqrt(2*nmodes) and sigma_quad = Pl2 + Pn/np.sqrt(2*nmodes)
        print(self.Pn)
        P_n_bin = self.Pn * np.ones(np.shape(k_eff)) 
        
        for kk in range(len(k_eff)):
            sigma_perp, sigma_par, resfac = get_resfac(Survey.z, Survey.nu_rest/(1+Survey.z), Survey.sigma_b, Survey.delta_nu, k = k_eff[kk])
#             P_k_bin_mono[kk] = expvals_mono[kk]*resfac
            P_n_bin[kk] = P_n_bin[kk]*resfac
        
        
        P_n_bin = P_n_bin + self.shot * np.ones(np.shape(k_eff)) 

#         def integrand(mu, kk):
#             return eval_legendre(2, mu)*eval_legendre(0, mu)*((1 + fsigma8*mu**2)**2*bI2*expvals[kk] + P_n_bin[kk])**2
        
#         Ilist = []
        
#         for kk in range(len(k_eff)): 
#             Ilist.append(quad(integrand, -1, 1, args = kk)[0])
    
#         Ilist = (2*2 + 1)*(2*0+1)*Ilist
#         interlopers = np.zeros(len(Survey.kList))
        
#         for i in range(0, len(Survey.bList)):
#             interlopers = interlopers + self.interloper(Survey, Modified_Gravity_Model, Survey.bList[i], Survey.zList[i])
        
#         interlopers_Pk, k_eff = get_expvals(Survey.k_bin, Survey.dlogk, Survey.kList, interlopers, bintype = 'log')
#         interlopers_quad = (4/3*fsigma8 + 4/7*fsigma8**2)*bI2*interlopers_Pk

        # prefactor = 2
        # first_term = (8/3*fsigma8 + 24/7 * fsigma8**2 + 40/21 * fsigma8**3 + 40/99*fsigma8**4)*(bI2*expvals)**2 
        # second_term = 2*P_n_bin*(4/3*fsigma8 + 4/7*fsigma8**2)*(bI2*expvals)
        # numerator = prefactor*(first_term + second_term)
        
        pre_factor = 2
        
        # check factor of f vs beta here
        
        first_term = (1 + 4/3*Beta + 6/5*Beta**2 + 4/7*Beta**3 + 1/9*Beta**4)*(bI2*expvals)**2
        second_term = 2*P_n_bin*(1+2/3*Beta + 1/5 * Beta**2)*bI2*expvals + P_n_bin**2
        
        numerator = pre_factor*(first_term + second_term)
    
        var_RSD_mono = numerator/nmodes
        
        prefactor = 2
        first_term = (5 + 220/21 * Beta + 90/7 * Beta**2 + 1700/231 * Beta**3 + 2075/1287 * Beta**4)*(bI2*expvals)**2
        second_term = 2*P_n_bin*(5 + 220/21*Beta + 30/7 * Beta**2)*(bI2*expvals)
        third_term = 5*P_n_bin**2 
    
        numerator = prefactor*(first_term + second_term + third_term) 
        
        var_RSD_quad = numerator/nmodes
        
        prefactor = 2
        first_term = (8/3*Beta + 24/7 * Beta**2 + 40/21 * Beta**3 + 40/99*Beta**4)*(bI2*expvals)**2 
        second_term = 2*P_n_bin*(4/3*Beta + 4/7*Beta**2)*(bI2*expvals)
        numerator = prefactor*(first_term + second_term)
        
        var_RSD_cov = numerator/nmodes
        
        C00_I_list = []
        C22_I_list = []
        C20_I_list = []
        
        for i in range(0, len(P_n_bin)):
            cov_matrix = np.mat([[var_RSD_mono[i], var_RSD_cov[i]], [var_RSD_cov[i], var_RSD_quad[i]]]).I
            C00_I_list.append(cov_matrix[0,0])
            C22_I_list.append(cov_matrix[1,1])
            C20_I_list.append(cov_matrix[1,0])
        
        return C00_I_list, C22_I_list, C20_I_list
    
###################################################### 8. All Combined with Interlopers ######################################################################
   
    def combined_cov(self, Survey, Modified_Gravity_Model):
        
        """
        Computes the Redshift Space Distortion (RSD) quadrupole errors *with* interlopers included. Returns an array containing the noise at the bin centers. 
        Included Noise components:
        1. Pn "white"
        2. Finite Instrument Spectral Resolution
        3. interlopers
        
        Parameters:
        Survey (Survey object): Survey/experiment object holding survey definition.
        Modified_Gravity_Model (Modified Gravity Model object): Modified Gravity model object holding the model definition
        
        """
        
        b = (1+Survey.z)
        bI = (1+Survey.z)*self.I
        bI2 = bI*bI
        
        nmodes = Survey.nmodes()
        
        fsigma8 = Modified_Gravity_Model.compute_fsigma8(Survey.z)
        Beta = Modified_Gravity_Model.f/b
        
        expvals, k_eff = get_expvals(Survey.k_bin, Survey.dlogk, Survey.kList, Modified_Gravity_Model.Pk, bintype = 'log')
        self.k_eff = k_eff
            
        P_n_bin = self.Pn * np.ones(np.shape(k_eff)) 
        
        for kk in range(len(k_eff)):
            sigma_perp, sigma_par, resfac = get_resfac(Survey.z, Survey.nu_rest/(1+Survey.z), Survey.sigma_b, Survey.delta_nu, k = k_eff[kk])
#             P_k_bin_mono[kk] = expvals_mono[kk]*resfac
            P_n_bin[kk] = P_n_bin[kk]*resfac
    
        for kk,k in enumerate(k_eff):
            kstol = k*cosmo.comoving_distance(0.5)/cosmo.h
            P_n_bin[kk] = P_n_bin[kk]*( 1 + (np.array(kstol)/self.l_knee)**(self.alpha))
        
        # + foregrounds
        # for kk, k in enumerate(k_eff):
        #     P_n_bin[kk] = P_n_bin[kk] + expvals[kk]*(1 + (k/self.eta_min)**self.Beta)
        
        
        P_n_bin = P_n_bin + self.shot * np.ones(np.shape(k_eff)) 
        
        pre_factor = 2
        
        
        # interloper_terms 
        
        interlopers = np.zeros(len(Survey.kList))
        interlopers_var_00 = np.zeros(len(k_eff))
        interlopers_var_22 = np.zeros(len(k_eff))
        interlopers_var_02 = np.zeros(len(k_eff))
        
        for i in range(0, len(Survey.bList)):
            fsigma8 = Modified_Gravity_Model.compute_fsigma8(Survey.zList[i])
            Beta_int = Modified_Gravity_Model.f/b
            interlopers = self.interloper(Survey, Modified_Gravity_Model, Survey.bList[i], Survey.zList[i])
            
            interlopers_Pk, k_eff_int = get_expvals(Survey.k_bin, Survey.dlogk, Survey.kList, interlopers, bintype = 'log')
        
            ## 00 
            
            first_term = (1 + 4/3*Beta_int + 6/5*Beta_int**2 + 4/7*Beta_int**3 + 1/9*Beta_int**4)*(interlopers_Pk)**2
            second_term = 2*0*(1+2/3*Beta_int + 1/5 * Beta_int**2)*interlopers_Pk + 0**2
            third_term = 5*0**2 
            
            interlopers_var_00 = interlopers_var_00 + pre_factor*(first_term + second_term + third_term) 
            
            ## 22
            first_term = (5 + 220/21 * Beta_int + 90/7 * Beta_int**2 + 1700/231 * Beta_int**3 + 2075/1287 * Beta_int**4)*(interlopers_Pk)**2
            second_term = 2*0*(5 + 220/21*Beta_int + 30/7 * Beta_int**2)*(interlopers_Pk)
            third_term = 5*0**2 
            
            interlopers_var_22 = interlopers_var_22 + pre_factor*(first_term + second_term + third_term) 
            
            ## 02 
            
            first_term = (8/3*Beta_int + 24/7 * Beta_int**2 + 40/21 * Beta_int**3 + 40/99*Beta_int**4)*(interlopers_Pk)**2 
            second_term = 2*0*(4/3*Beta_int + 4/7*Beta_int**2)*(interlopers_Pk)
            interlopers_var_02 = interlopers_var_02 + pre_factor*(first_term + second_term)
        
        # C00
        # check factor of f vs beta here
        
        pre_factor = 2
        
        first_term = (1 + 4/3*Beta + 6/5*Beta**2 + 4/7*Beta**3 + 1/9*Beta**4)*(bI2*expvals)**2
        second_term = 2*P_n_bin*(1+2/3*Beta + 1/5 * Beta**2)*bI2*expvals + P_n_bin**2
        
        numerator = pre_factor*(first_term + second_term)
    
        var_RSD_mono = (numerator + interlopers_var_00 + P_n_bin)/nmodes
        
        # C22 
        
        first_term = (5 + 220/21 * Beta + 90/7 * Beta**2 + 1700/231 * Beta**3 + 2075/1287 * Beta**4)*(bI2*expvals)**2
        second_term = 2*P_n_bin*(5 + 220/21*Beta + 30/7 * Beta**2)*(bI2*expvals)
        third_term = 5*P_n_bin**2 
    
        numerator = pre_factor*(first_term + second_term + third_term) 
        
        var_RSD_quad = (numerator + interlopers_var_22 + P_n_bin)/nmodes
        
        # C02, C20
            
        first_term = (8/3*Beta + 24/7 * Beta**2 + 40/21 * Beta**3 + 40/99*Beta**4)*(bI2*expvals)**2 
        second_term = 2*P_n_bin*(4/3*Beta + 4/7*Beta**2)*(bI2*expvals)
        numerator = pre_factor*(first_term + second_term)
        
        var_RSD_cov = (numerator + interlopers_var_02 + P_n_bin)/nmodes
        
        
        C00_I_list = []
        C22_I_list = []
        C20_I_list = []

        
        for i in range(0, len(P_n_bin)):
            cov_matrix = np.mat([[var_RSD_mono[i], var_RSD_cov[i]], [var_RSD_cov[i], var_RSD_quad[i]]]).I
            C00_I_list.append(cov_matrix[0,0])
            C22_I_list.append(cov_matrix[1,1])
            C20_I_list.append(cov_matrix[1,0])
        
        return C00_I_list, C22_I_list, C20_I_list