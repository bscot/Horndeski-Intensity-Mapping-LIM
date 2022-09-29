#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from astropy.cosmology import Planck15 as cosmo
from scipy.stats import chi2, norm
import scipy.special as spec

from classy import Class
from scipy.optimize import curve_fit

from bryan_im_utils import *


# In[2]:


from bryan_fisher_tools import *


# In[3]:


# from MG_IM_models_no_atmosphere_with_RSD import *

from MG_IM_models import *

print('imports complete')


# In[4]:


k_max = 5e-2
k_min = -3
bins = 20

# nu_rest = 231
# B = 50
# z_in = 0.5
# D = 10.0 #m Dish size
# theta_b = 1.2*(2.998e8 / ((nu_rest/(1+z_in))*1e9)) # FWHM in radians
# sigma_b = theta_b/2.35 #Gaussian sigma for beam, radians

# field_length = 128 # deg # sky fraction from Chili, 40% --> sqrt(0.4*4 pi Sr * (180/pi)**2) --> Sr --> deg
# field_length = 45
delta_nu = 0.5 #500 Mhz bins

Pn = 480
alpha = -2.8 #-5/3
Beta = -5/3
l_knee = 200 #72.9
eta_min = 0.05

k_List = np.logspace(-5, k_max, 3000)
kList = np.logspace(-5, k_max, 3000) #fix this

z_10_150 = (115. / 150.0) - 1.0 # z < 0
z_21_150 = (230. / 150.0) - 1.0
z_32_150 = (345. / 150.0) - 1.0
z_43_150 = (461. / 150.0) - 1.0
z_54_150 = (576. / 150.0) - 1.0
z_65_150 = (691. / 150.0) - 1.0
z_76_150 = (806. / 150.0) - 1.0
z_98_150 = (1036. / 150.0) - 1.0

nu_10 = 115
nu_21 = 230
nu_32 = 345
nu_43 = 461
nu_54 = 576
nu_65 = 691
nu_76 = 806
nu_98 = 1036

# b_21_150 = 0.2238
# b_32_150 = 0.5008
# b_43_150 = 0.2128
# b_54_150 = 0.0963388
# b_65_150 = 0.04094
# b_76_150 = 0.01634
# b_98_150 = 0.0036565

# b_21_150 = 0.2238
# b_32_150 = 0.5008
# b_43_150 = 0.2128
# b_54_150 = 0.0963388
# b_65_150 = 0.04094
# b_76_150 = 0.01634
# b_98_150 = 0.0036565

z_10_90 = (115. / 95.0) - 1.0
z_21_90 = (230. / 95.0) - 1.0
z_32_90 = (345. / 95.0) - 1.0
z_43_90 = (461. / 95.0) - 1.0
z_54_90 = (576. / 95.0) - 1.0
z_65_90 = (691. / 95.0) - 1.0
z_76_90 = (806. / 95.0) - 1.0
#z_98_90 = (1036. / 95.0) - 1.0

# b_10_90 = 0.098
# b_21_90 = 0.71769
# b_32_90 = 0.617
# b_43_90 = 0.12647
# b_54_90 = 0.04179
# b_65_90 = 0.01566
# b_76_90 = 0.006

z_10_245 = (115. / 245.0) - 1.0
z_21_245 = (230. / 245.0) - 1.0
z_32_245 = (345. / 245.0) - 1.0
z_43_245 = (461. / 245.0) - 1.0
z_54_245 = (576. / 245.0) - 1.0
z_65_245 = (691. / 245.0) - 1.0
z_76_245 = (806. / 245.0) - 1.0
z_98_245 = (1036. / 245.0) - 1.0


#b21_245 =
# b_32_245 = 0.1326
# b_43_245 = 0.0993
# b_54_245 = 0.08868
# b_65_245 = 0.068089
# b_76_245 = 0.0358
# b_98_245 = 0.01044

# updated band-averaged values. 

b_10_150, b_10_245, b_10_90 = [0, 0, 0.14125974512653172]
b_21_150, b_21_245, b_21_90 = [0.23814300654524978, 0.05993319334046567, 0.7518435116225263]
b_32_150, b_32_245, b_32_90 = [0.4592268867263453, 0.14029340028678336, 0.6030114302796646]
b_43_150, b_43_245, b_43_90 = [0.206084638401821, 0.0945815471506361, 0.0960524943669473]
b_54_150, b_54_245, b_54_90 = [0.0982545516195774, 0.085691926555859, 0.0343092085605491]
b_65_150, b_65_245, b_65_90 = [0.03998000363105247, 0.06354885042666927, 0.01271079551252695]
b_76_150, b_76_245, b_76_90 = [0.015828137953894098, 0.0346042360296219, 0.00454225728321831]
b_87_150, b_87_245, b_87_90 = [0.00698181582124012, 0.018496707188195648, 0]
b_98_150, b_98_245, b_98_90 = [0.0037430074500111703, 0.01025706724259277, 0]

# shot noise values

I10 = 2.65 #498
I21 = 8.54 # 498
I32_04 = 2.97 #226
I32_13 = 81.7
I32_14 = 100
I32_26 = 295 
I43 = 91
I54 = 35
CII = 21
I65 = 9


# In[5]:


def f_Pk_DE(k_eff, Survey, model, *args):

    """
    Observable function to pass to the fisher matrix methods. Returns power spectra assuming the propto DE model. 

    Parameters:
    Survey (survey object): object holding survey definition attributes. 
    model (Modified gravity model object): object holding modified gravity model definition. 
    args: values of modified gravity model slope parameters in {x_k, x_b, x_m, x_t} order. Only x_b, x_m will be used. 

    Returns:
    numpy array: holding the binned power spectra multiplied by bI^2

    """
    
    x_k = 1
    x_b = list(args[0])[0] #have been using 0.6, should make sure all are rerun to 0.63 standard 
    x_m = list(args[0])[1]
    x_t = 0
    M = 1
    
    bI21 = (1+Survey.z)*Survey.I
    bI21 = bI21**2
    
    MG_model = Modified_Gravity_Model()

    MG_model.set_model('DE', x_k, x_b, x_m, x_t, M)
    Pks = MG_model.compute_MG_Pk(Survey.z, kList)
    
    MG_expvals, k_eff = get_expvals(Survey.k_bin, Survey.dlogk, Survey.kList, Pks, bintype = 'log')
    
    return bI21 * MG_expvals

def f_Pk_a(k_eff, Survey, model, *args):

    """
    Observable function to pass to the fisher matrix methods. Returns power spectra assuming the propto a model.

    Parameters:
    Survey (survey object): object holding survey definition attributes. 
    model (Modified gravity model object): object holding modified gravity model definition. 
    args: values of modified gravity model slope parameters in {x_k, x_b, x_m, x_t} order. Only x_b, x_m will be used. 

    Returns:
    numpy array: holding the binned power spectra multiplied by bI^2

    """
    
    
    x_k = 1
    x_b = list(args[0])[0] #have been using 0.6, should make sure all are rerun to 0.63 standard 
    x_m = list(args[0])[1]
    x_t = 0
    M = 1
    
    bI21 = (1+Survey.z)*Survey.I
    bI21 = bI21**2
    
    MG_model = Modified_Gravity_Model()

    MG_model.set_model('a', x_k, x_b, x_m, x_t, M)
    Pks = MG_model.compute_MG_Pk(Survey.z, kList)
    
    MG_expvals, k_eff = get_expvals(Survey.k_bin, Survey.dlogk, Survey.kList, Pks, bintype = 'log')
    
    return bI21 * MG_expvals

def f_Pk_RSD_a_mono(k_eff, Survey, model, *args):
    
    """
    Observable function to pass to the fisher matrix methods. Returns RSD monopole assuming the propto a model.

    Parameters:
    Survey (survey object): object holding survey definition attributes. 
    model (Modified gravity model object): object holding modified gravity model definition. 
    args: values of modified gravity model slope parameters in {x_k, x_b, x_m, x_t} order. Only x_b, x_m will be used. 

    Returns:
    numpy array: holding the binned monopole. 

    """
    
    
    b = (1+Survey.z)
    bI = (1+Survey.z)*Survey.I
    bI2 = bI*bI
    
    x_k = 1
    x_b = list(args[0])[0] #have been using 0.6, should make sure all are rerun to 0.63 standard 
    x_m = list(args[0])[1]
    x_t = 0
    M = 1
    
    bI21 = (1+Survey.z)*Survey.I
    bI21 = bI21**2
    
    MG_model = Modified_Gravity_Model()

    MG_model.set_model('a', x_k, x_b, x_m, x_t, M)
    Pks = MG_model.compute_MG_Pk(Survey.z, kList)
    fsigma8 = MG_model.compute_fsigma8(Survey.z)
    Beta = MG_model.f/b
    
    MG_expvals, k_eff = get_expvals(Survey.k_bin, Survey.dlogk, Survey.kList, Pks, bintype = 'log')
    
    # fix this - needs to return *fsigma8
    
    return (1 + 2/3*Beta + 1/5*Beta**2)*bI2*MG_expvals #update with fsigma8

def f_Pk_RSD_a_quad(k_eff, Survey, model, *args):
    
    """
    Observable function to pass to the fisher matrix methods. Returns RSD quadrupole assuming the propto a model.

    Parameters:
    Survey (survey object): object holding survey definition attributes. 
    model (Modified gravity model object): object holding modified gravity model definition. 
    args: values of modified gravity model slope parameters in {x_k, x_b, x_m, x_t} order. Only x_b, x_m will be used. 

    Returns:
    numpy array: holding the binned RSD quadrupole.

    """
    
    
    b = (1+Survey.z)
    bI = (1+Survey.z)*Survey.I
    bI2 = bI*bI
    
    x_k = 1
    x_b = list(args[0])[0] #have been using 0.6, should make sure all are rerun to 0.63 standard 
    x_m = list(args[0])[1]
    x_t = 0
    M = 1
    
    bI21 = (1+Survey.z)*Survey.I
    bI21 = bI21**2
    
    MG_model = Modified_Gravity_Model()

    MG_model.set_model('a', x_k, x_b, x_m, x_t, M)
    Pks = MG_model.compute_MG_Pk(Survey.z, kList)
    fsigma8 = MG_model.compute_fsigma8(Survey.z)
    Beta = MG_model.f/b
    MG_expvals, k_eff = get_expvals(Survey.k_bin, Survey.dlogk, Survey.kList, Pks, bintype = 'log')
    
    # fix this - needs to return *fsigma8
    
    return (4/3*Beta + 4/7*Beta**2)*bI2*MG_expvals #update with fsigma8


def f_Pk_RSD_DE_mono(k_eff, Survey, model, *args):
    
    """
    Observable function to pass to the fisher matrix methods. Returns RSD quadrupole assuming the propto DE model.

    Parameters:
    Survey (survey object): object holding survey definition attributes. 
    model (Modified gravity model object): object holding modified gravity model definition. 
    args: values of modified gravity model slope parameters in {x_k, x_b, x_m, x_t} order. Only x_b, x_m will be used. 

    Returns:
    numpy array: holding the binned RSD monopole.

    """
    
    
    b = (1+Survey.z)
    bI = (1+Survey.z)*Survey.I
    bI2 = bI*bI
    
    x_k = 1
    x_b = list(args[0])[0] #have been using 0.6, should make sure all are rerun to 0.63 standard 
    x_m = list(args[0])[1]
    x_t = 0
    M = 1
    
    bI21 = (1+Survey.z)*Survey.I
    bI21 = bI21**2
    
    MG_model = Modified_Gravity_Model()

    MG_model.set_model('DE', x_k, x_b, x_m, x_t, M)
    Pks = MG_model.compute_MG_Pk(Survey.z, kList)
    fsigma8 = MG_model.compute_fsigma8(Survey.z)
    Beta = MG_model.f/b
    
    MG_expvals, k_eff = get_expvals(Survey.k_bin, Survey.dlogk, Survey.kList, Pks, bintype = 'log')
    
    # fix this - needs to return *fsigma8
    
    return (1 + 2/3*Beta + 1/5*Beta**2)*bI2*MG_expvals #update with fsigma8

def f_Pk_RSD_DE_quad(k_eff, Survey, model, *args):
    
    """
    Observable function to pass to the fisher matrix methods. Returns RSD quadrupole assuming the propto DE model.

    Parameters:
    Survey (survey object): object holding survey definition attributes. 
    model (Modified gravity model object): object holding modified gravity model definition. 
    args: values of modified gravity model slope parameters in {x_k, x_b, x_m, x_t} order. Only x_b, x_m will be used. 

    Returns:
    numpy array: holding the binned RSD quadrupole.

    """
    
    b = (1+Survey.z)
    bI = (1+Survey.z)*Survey.I
    bI2 = bI*bI
    
    x_k = 1
    x_b = list(args[0])[0] #have been using 0.6, should make sure all are rerun to 0.63 standard 
    x_m = list(args[0])[1]
    x_t = 0
    M = 1
    
    bI21 = (1+Survey.z)*Survey.I
    bI21 = bI21**2
    
    MG_model = Modified_Gravity_Model()

    MG_model.set_model('DE', x_k, x_b, x_m, x_t, M)
    Pks = MG_model.compute_MG_Pk(Survey.z, kList)
    fsigma8 = MG_model.compute_fsigma8(Survey.z)
    Beta = MG_model.f/b
    
    MG_expvals, k_eff = get_expvals(Survey.k_bin, Survey.dlogk, Survey.kList, Pks, bintype = 'log')
    
    # fix this - needs to return *fsigma8
    
    return (4/3*Beta + 4/7*Beta**2)*bI2*MG_expvals #update with fsigma8

# In[6]:


def run_forecast_mono(Pn, P_shot, B_tar, z_tar, nu_rest_tar, b_in_list, z_in_list, line_intensity_tar, res, band_name, out_file, model_selection, field_length):
    
    """
    Method to define/update parameters of survey and noise models, then run the forecast code and return the result. This method runs the RSD monopole model. Probably redundant. 

    Parameters:
    Pn (float) : "White" (additive) noise level - units are set by the units of bI^2 and the detector time conversion. 
    B_tar (float) = Target Bandwidth - set by the presence of strong atmospheric emission lines and detector physics
    z_tar (float) = Target redshift
    nu_rest_tar (float) = Target emission line restframe frequency
    b_in_list (array/list) = interloper emission line intensities 
    z_in_list (array/list) = interloper emission line redshifts
    line_intensity_tar (float) = target emission line intensity - units should match/set those of Pn
    band_name (str) = bandwidth name for appending to out_file
    out_file (str) = out filename 
    model_selection (str) = Modified gravity evolution parameter choice, either 'a' or 'DE' for propto scale factor and dark     energy density, respectively 

    Returns:
    numpy array: Holding fisher matrix for input parameters. 

    """  
    
    B = B_tar
    z_target = z_tar
    z_in = z_tar
    nu_rest = nu_rest_tar
    
    D = 10.0 #m Dish size
    theta_b = 1.2*(2.998e8 / ((nu_rest/(1+z_in))*1e9)) # FWHM in radians
    sigma_b = theta_b/2.35 #Gaussian sigma for beam, radians
    
    b_list = b_in_list
    z_list = z_in_list
    print('resolution is: ')
    print(res)
    
    Survey1 = Survey(B, D, theta_b, field_length, res, k_List)
    Survey1.set_targets(z_target, nu_rest, line_intensity_tar, k_min, k_max, bins, b_list, z_list)
    Survey1.set_survey_Volume()
    print(Survey1.V_s)

    kList = Survey1.kList
    k_bin = Survey1.k_bin

    target_line_intensity = line_intensity_tar

    MG_model = Modified_Gravity_Model()
    
    if model_selection == 'DE':

        MG_model.set_model('DE', 1, 0.63, 0.2, 0, 1) #change to set DE or a model
        MG_model.compute_MG_Pk(z_target, kList)
    
        noises = Noise_Model(Pn, P_shot, alpha, l_knee, Beta, eta_min, target_line_intensity)
        noises.set_Noise_model('RSD_mono', Survey1, MG_model)

        noises_test = noises.sigma
    
        proptoDE_fisher = fisher(f_Pk_RSD_DE_mono, Survey1, 'DE', noises.k_eff, 0.01, noises_test, 0.63, 0.2)
        
    elif model_selection == 'a':
        
        MG_model.set_model('a', 1, 0.48, 0.27, 0, 1) #change to set DE or a model
        MG_model.compute_MG_Pk(z_target, kList)
    
        noises = Noise_Model(Pn, P_shot, alpha, l_knee, Beta, eta_min, target_line_intensity)
        noises.set_Noise_model('RSD_mono', Survey1, MG_model)

        noises_test = noises.sigma
    
        proptoDE_fisher = fisher(f_Pk_RSD_a_mono, Survey1, 'a', noises.k_eff, 0.01, noises_test, 0.48, 0.27)
        
    else:
        raise ValueError('Modified Gravity Model must be a or DE.')

        
    
    return proptoDE_fisher

def run_forecast_quad(Pn, P_shot, B_tar, z_tar, nu_rest_tar, b_in_list, z_in_list, line_intensity_tar, res, band_name, out_file, model_selection, field_length):
    
    """
    Method to define/update parameters of survey and noise models, then run the forecast code and return the result. This method runs the quadrupole error model. Probably redundant. 

    Parameters:
    Pn (float) : "White" (additive) noise level - units are set by the units of bI^2 and the detector time conversion. 
    B_tar (float) = Target Bandwidth - set by the presence of strong atmospheric emission lines and detector physics
    z_tar (float) = Target redshift
    nu_rest_tar (float) = Target emission line restframe frequency
    b_in_list (array/list) = interloper emission line intensities 
    z_in_list (array/list) = interloper emission line redshifts
    line_intensity_tar (float) = target emission line intensity - units should match/set those of Pn
    band_name (str) = bandwidth name for appending to out_file
    out_file (str) = out filename 
    model_selection (str) = Modified gravity evolution parameter choice, either 'a' or 'DE' for propto scale factor and dark     energy density, respectively 

    Returns:
    numpy array: Holding fisher matrix for input parameters. 

    """  
    
    B = B_tar
    z_target = z_tar
    z_in = z_tar
    nu_rest = nu_rest_tar
    
    D = 10.0 #m Dish size
    theta_b = 1.2*(2.998e8 / ((nu_rest/(1+z_in))*1e9)) # FWHM in radians
    sigma_b = theta_b/2.35 #Gaussian sigma for beam, radians
    
    b_list = b_in_list
    z_list = z_in_list
    print('resolution is: ')
    print(res)
    
    Survey1 = Survey(B, D, theta_b, field_length, res, k_List)
    Survey1.set_targets(z_target, nu_rest, line_intensity_tar, k_min, k_max, bins, b_list, z_list)
    Survey1.set_survey_Volume()
    print(Survey1.V_s)

    kList = Survey1.kList
    k_bin = Survey1.k_bin

    target_line_intensity = line_intensity_tar

    MG_model = Modified_Gravity_Model()


    if model_selection == 'DE':

        MG_model.set_model('DE', 1, 0.63, 0.2, 0, 1) #change to set DE or a model
        MG_model.compute_MG_Pk(z_target, kList)
    
        noises = Noise_Model(Pn, P_shot, alpha, l_knee, Beta, eta_min, target_line_intensity)
        noises.set_Noise_model('RSD_quad', Survey1, MG_model)

        noises_test = noises.sigma
    
        proptoDE_fisher = fisher(f_Pk_RSD_DE_quad, Survey1, 'DE', noises.k_eff, 0.01, noises_test, 0.63, 0.2)
        
    elif model_selection == 'a':
        
        MG_model.set_model('a', 1, 0.48, 0.27, 0, 1) #change to set DE or a model
        MG_model.compute_MG_Pk(z_target, kList)
    
        noises = Noise_Model(Pn, P_shot, alpha, l_knee, Beta, eta_min, target_line_intensity)
        noises.set_Noise_model('RSD_quad', Survey1, MG_model)

        noises_test = noises.sigma
    
        proptoDE_fisher = fisher(f_Pk_RSD_a_quad, Survey1, 'a', noises.k_eff, 0.01, noises_test, 0.48, 0.27)
        
    else:
        raise ValueError('Modified Gravity Model must be a or DE.')
        
    
    return proptoDE_fisher

def run_forecast_mono_interloper(Pn, P_shot, B_tar, z_tar, nu_rest_tar, b_in_list, z_in_list, line_intensity_tar, res, band_name, out_file, model_selection, field_length):
    
    B = B_tar
    z_target = z_tar
    z_in = z_tar
    nu_rest = nu_rest_tar
    
    D = 10.0 #m Dish size
    theta_b = 1.2*(2.998e8 / ((nu_rest/(1+z_in))*1e9)) # FWHM in radians
    sigma_b = theta_b/2.35 #Gaussian sigma for beam, radians
    
    b_list = b_in_list
    z_list = z_in_list
    
    Survey1 = Survey(B, D, theta_b, field_length, res, k_List)
    Survey1.set_targets(z_target, nu_rest, line_intensity_tar, k_min, k_max, bins, b_list, z_list)
    Survey1.set_survey_Volume()
    print(Survey1.V_s)

    kList = Survey1.kList
    k_bin = Survey1.k_bin

    target_line_intensity = line_intensity_tar

    MG_model = Modified_Gravity_Model()
    
    if model_selection == 'DE':

        MG_model.set_model('DE', 1, 0.63, 0.2, 0, 1) #change to set DE or a model
        MG_model.compute_MG_Pk(z_target, kList)
    
        noises = Noise_Model(Pn, P_shot, alpha, l_knee, Beta, eta_min, target_line_intensity)
        noises.set_Noise_model('RSD_mono_interloper', Survey1, MG_model)

        noises_test = noises.sigma
    
        proptoDE_fisher = fisher(f_Pk_RSD_DE_mono, Survey1, 'DE', noises.k_eff, 0.01, noises_test, 0.63, 0.2)
        
    elif model_selection == 'a':
        
        MG_model.set_model('a', 1, 0.48, 0.27, 0, 1) #change to set DE or a model
        MG_model.compute_MG_Pk(z_target, kList)
    
        noises = Noise_Model(Pn, P_shot, alpha, l_knee, Beta, eta_min, target_line_intensity)
        noises.set_Noise_model('RSD_mono_interloper', Survey1, MG_model)

        noises_test = noises.sigma
    
        proptoDE_fisher = fisher(f_Pk_RSD_a_mono, Survey1, 'a', noises.k_eff, 0.01, noises_test, 0.48, 0.27)
        
    else:
        raise ValueError('Modified Gravity Model must be a or DE.')

        
    
    return proptoDE_fisher

def run_forecast_quad_interloper(Pn, P_shot, B_tar, z_tar, nu_rest_tar, b_in_list, z_in_list, line_intensity_tar, res, band_name, out_file, model_selection, field_length):
    
    B = B_tar
    z_target = z_tar
    z_in = z_tar
    nu_rest = nu_rest_tar
    
    D = 10.0 #m Dish size
    theta_b = 1.2*(2.998e8 / ((nu_rest/(1+z_in))*1e9)) # FWHM in radians
    sigma_b = theta_b/2.35 #Gaussian sigma for beam, radians
    
    b_list = b_in_list
    z_list = z_in_list
    
    Survey1 = Survey(B, D, theta_b, field_length, res, k_List)
    Survey1.set_targets(z_target, nu_rest, line_intensity_tar, k_min, k_max, bins, b_list, z_list)
    Survey1.set_survey_Volume()
    print(Survey1.V_s)

    kList = Survey1.kList
    k_bin = Survey1.k_bin

    target_line_intensity = line_intensity_tar

    MG_model = Modified_Gravity_Model()


    if model_selection == 'DE':

        MG_model.set_model('DE', 1, 0.63, 0.2, 0, 1) #change to set DE or a model
        MG_model.compute_MG_Pk(z_target, kList)
    
        noises = Noise_Model(Pn, P_shot, alpha, l_knee, Beta, eta_min, target_line_intensity)
        noises.set_Noise_model('RSD_quad_interloper', Survey1, MG_model)

        noises_test = noises.sigma
    
        proptoDE_fisher = fisher(f_Pk_RSD_DE_quad, Survey1, 'DE', noises.k_eff, 0.01, noises_test, 0.63, 0.2)
        
    elif model_selection == 'a':
        
        MG_model.set_model('a', 1, 0.48, 0.27, 0, 1) #change to set DE or a model
        MG_model.compute_MG_Pk(z_target, kList)
    
        noises = Noise_Model(Pn, P_shot, alpha, l_knee, Beta, eta_min, target_line_intensity)
        noises.set_Noise_model('RSD_quad_interloper', Survey1, MG_model)

        noises_test = noises.sigma
    
        proptoDE_fisher = fisher(f_Pk_RSD_a_quad, Survey1, 'a', noises.k_eff, 0.01, noises_test, 0.48, 0.27)
        
    else:
        raise ValueError('Modified Gravity Model must be a or DE.')
        
    
    return proptoDE_fisher

def run_forecast_fiducial_mitigated(Pn, P_shot, B_tar, z_tar, nu_rest_tar, b_in_list, z_in_list, line_intensity_tar, res, band_name, out_file, model_selection, field_length):
    
    B = B_tar
    z_target = z_tar
    z_in = z_tar
    nu_rest = nu_rest_tar
    
    D = 10.0 #m Dish size
    theta_b = 1.2*(2.998e8 / ((nu_rest/(1+z_in))*1e9)) # FWHM in radians
    sigma_b = theta_b/2.35 #Gaussian sigma for beam, radians
    
    b_list = b_in_list
    z_list = z_in_list
    
    Survey1 = Survey(B, D, theta_b, field_length, res, k_List)
    Survey1.set_targets(z_target, nu_rest, line_intensity_tar, k_min, k_max, bins, b_list, z_list)
    Survey1.set_survey_Volume()
    print(Survey1.V_s)

    kList = Survey1.kList
    k_bin = Survey1.k_bin

    target_line_intensity = line_intensity_tar

    MG_model = Modified_Gravity_Model()


    if model_selection == 'DE':

        MG_model.set_model('DE', 1, 0.63, 0.2, 0, 1) #change to set DE or a model
        MG_model.compute_MG_Pk(z_target, kList)
    
        noises = Noise_Model(Pn, P_shot, alpha, l_knee, Beta, eta_min, target_line_intensity)
        noises.set_Noise_model('fiducial_mitigated', Survey1, MG_model)

        noises_test = noises.sigma
    
        proptoDE_fisher = fisher(f_Pk_DE, Survey1, 'DE', noises.k_eff, 0.01, noises_test, 0.63, 0.2)
        
    elif model_selection == 'a':
        
        MG_model.set_model('a', 1, 0.48, 0.27, 0, 1) #change to set DE or a model
        MG_model.compute_MG_Pk(z_target, kList)
    
        noises = Noise_Model(Pn, P_shot, alpha, l_knee, Beta, eta_min, target_line_intensity)
        noises.set_Noise_model('fiducial_mitigated', Survey1, MG_model)

        noises_test = noises.sigma
    
        proptoDE_fisher = fisher(f_Pk_a, Survey1, 'a', noises.k_eff, 0.01, noises_test, 0.48, 0.27)
        
    else:
        raise ValueError('Modified Gravity Model must be a or DE.')
        
    
    return proptoDE_fisher

def run_forecast_fiducial_unmitigated(Pn, P_shot, B_tar, z_tar, nu_rest_tar, b_in_list, z_in_list, line_intensity_tar, res, band_name, out_file, model_selection, field_length):
    
    B = B_tar
    z_target = z_tar
    z_in = z_tar
    nu_rest = nu_rest_tar
    
    D = 10.0 #m Dish size
    theta_b = 1.2*(2.998e8 / ((nu_rest/(1+z_in))*1e9)) # FWHM in radians
    sigma_b = theta_b/2.35 #Gaussian sigma for beam, radians
    
    b_list = b_in_list
    z_list = z_in_list
    
    Survey1 = Survey(B, D, theta_b, field_length, res, k_List)
    Survey1.set_targets(z_target, nu_rest, line_intensity_tar, k_min, k_max, bins, b_list, z_list)
    Survey1.set_survey_Volume()
    print(Survey1.V_s)

    kList = Survey1.kList
    k_bin = Survey1.k_bin

    target_line_intensity = line_intensity_tar

    MG_model = Modified_Gravity_Model()


    if model_selection == 'DE':

        MG_model.set_model('DE', 1, 0.63, 0.2, 0, 1) #change to set DE or a model
        MG_model.compute_MG_Pk(z_target, kList)
    
        noises = Noise_Model(Pn, P_shot, alpha, l_knee, Beta, eta_min, target_line_intensity)
        noises.set_Noise_model('fiducial_unmitigated', Survey1, MG_model)

        noises_test = noises.sigma
    
        proptoDE_fisher = fisher(f_Pk_DE, Survey1, 'DE', noises.k_eff, 0.01, noises_test, 0.63, 0.2)
        
    elif model_selection == 'a':
        
        MG_model.set_model('a', 1, 0.48, 0.27, 0, 1) #change to set DE or a model
        MG_model.compute_MG_Pk(z_target, kList)
    
        noises = Noise_Model(Pn, P_shot, alpha, l_knee, Beta, eta_min, target_line_intensity)
        noises.set_Noise_model('fiducial_unmitigated', Survey1, MG_model)

        noises_test = noises.sigma
    
        proptoDE_fisher = fisher(f_Pk_a, Survey1, 'a', noises.k_eff, 0.01, noises_test, 0.48, 0.27)
        
    else:
        raise ValueError('Modified Gravity Model must be a or DE.')
        
    
    return proptoDE_fisher


def run_forecast_covariance(Pn, P_shot, B_tar, z_tar, nu_rest_tar, b_in_list, z_in_list, line_intensity_tar, res, band_name, out_file, model_selection, field_length):
    
    B = B_tar
    z_target = z_tar
    z_in = z_tar
    nu_rest = nu_rest_tar
    
    D = 10.0 #m Dish size
    theta_b = 1.2*(2.998e8 / ((nu_rest/(1+z_in))*1e9)) # FWHM in radians
    sigma_b = theta_b/2.35 #Gaussian sigma for beam, radians
    
    b_list = b_in_list
    z_list = z_in_list
    
    Survey1 = Survey(B, D, theta_b, field_length, res, k_List)
    Survey1.set_targets(z_target, nu_rest, line_intensity_tar, k_min, k_max, bins, b_list, z_list)
    Survey1.set_survey_Volume()
    print(Survey1.V_s)
    
    kList = Survey1.kList
    k_bin = Survey1.k_bin

    target_line_intensity = line_intensity_tar

    MG_model = Modified_Gravity_Model()


    if model_selection == 'DE':

        MG_model.set_model('DE', 1, 0.63, 0.2, 0, 1) #change to set DE or a model
        MG_model.compute_MG_Pk(z_target, kList)
    
        noises = Noise_Model(Pn, P_shot, alpha, l_knee, Beta, eta_min, target_line_intensity)
        noises.set_Noise_model('covariance', Survey1, MG_model)

        noise_C00 = noises.sigma[0]
        noise_C22 = noises.sigma[1]
        noise_C20 = noises.sigma[2]
    
        proptoDE_fisher = fisher_cov(f_Pk_RSD_DE_mono, f_Pk_RSD_DE_quad, Survey1, 'DE', noises.k_eff, 0.01, noise_C00, noise_C22, noise_C20, 0.63, 0.2)
        
    elif model_selection == 'a':
        
        MG_model.set_model('a', 1, 0.48, 0.27, 0, 1) #change to set DE or a model
        MG_model.compute_MG_Pk(z_target, kList)
    
        noises = Noise_Model(Pn, P_shot, alpha, l_knee, Beta, eta_min, target_line_intensity)
        noises.set_Noise_model('covariance', Survey1, MG_model)

        noise_C00 = noises.sigma[0]
        noise_C22 = noises.sigma[1]
        noise_C20 = noises.sigma[2]
    
        proptoDE_fisher = fisher_cov(f_Pk_RSD_a_mono, f_Pk_RSD_a_quad, Survey1, 'a', noises.k_eff, 0.01, noise_C00, noise_C22, noise_C20, 0.48, 0.27)
        
    else:
        raise ValueError('Modified Gravity Model must be a or DE.')
        
    
    return proptoDE_fisher

def run_forecast_covariance_full(Pn, P_shot, B_tar, z_tar, nu_rest_tar, b_in_list, z_in_list, line_intensity_tar, res, band_name, out_file, model_selection, field_length):
    
    B = B_tar
    z_target = z_tar
    z_in = z_tar
    nu_rest = nu_rest_tar
    
    D = 10.0 #m Dish size
    theta_b = 1.2*(2.998e8 / ((nu_rest/(1+z_in))*1e9)) # FWHM in radians
    sigma_b = theta_b/2.35 #Gaussian sigma for beam, radians
    
    b_list = b_in_list
    z_list = z_in_list
    
    Survey1 = Survey(B, D, theta_b, field_length, res, k_List)
    Survey1.set_targets(z_target, nu_rest, line_intensity_tar, k_min, k_max, bins, b_list, z_list)
    Survey1.set_survey_Volume()
    print(Survey1.V_s)

    kList = Survey1.kList
    k_bin = Survey1.k_bin

    target_line_intensity = line_intensity_tar

    MG_model = Modified_Gravity_Model()


    if model_selection == 'DE':

        MG_model.set_model('DE', 1, 0.63, 0.2, 0, 1) #change to set DE or a model
        MG_model.compute_MG_Pk(z_target, kList)
    
        noises = Noise_Model(Pn, P_shot, alpha, l_knee, Beta, eta_min, target_line_intensity)
        noises.set_Noise_model('covariance_full', Survey1, MG_model)

        noise_C00 = noises.sigma[0]
        noise_C22 = noises.sigma[1]
        noise_C20 = noises.sigma[2]
        
        # C00_I_list, C22_I_list, C20_I_list, Cmm_I_list, C0m_I_list, C2m_I_list
        # C00, C22, C20, C0m, C2m, Cmm,
        proptoDE_fisher = fisher_cov(f_Pk_RSD_DE_mono, f_Pk_RSD_DE_quad, Survey1, 'DE', noises.k_eff, 0.01, noise_C00, noise_C22, noise_C20, 0.63, 0.2)
        
        # (f, f2, f3 Survey, model, xval, step, C00, C22, C20, C0m, C2m, Cmm, *args)
        
    elif model_selection == 'a':
        
        MG_model.set_model('a', 1, 0.48, 0.27, 0, 1) #change to set DE or a model
        MG_model.compute_MG_Pk(z_target, kList)
    
        noises = Noise_Model(Pn, P_shot, alpha, l_knee, Beta, eta_min, target_line_intensity)
        noises.set_Noise_model('covariance_full', Survey1, MG_model)

        noise_C00 = noises.sigma[0]
        noise_C22 = noises.sigma[1]
        noise_C20 = noises.sigma[2]
    
        proptoDE_fisher = fisher_cov(f_Pk_RSD_a_mono, f_Pk_RSD_a_quad, Survey1, 'a', noises.k_eff, 0.01, noise_C00, noise_C22, noise_C20, 0.48, 0.27)
        
    else:
        raise ValueError('Modified Gravity Model must be a or DE.')
        
    
    return proptoDE_fisher


# In[7]:


def main():
    """
    main function of the script, sets and initializes various parameters, prints files to log etc. 
    
    This computes the Fisher matrix as the sum of each target line in each bandwidth window.
    
    The full forecast will use all target lines in all bandwidth windows, e.g. the sum of the sum of Fisher matrices calculated here. 
    
    """
    
    import sys

    if len(sys.argv) > 2: 
        print(len(sys.argv))
        print('Number of arguments:', len(sys.argv), 'arguments.')
        print('Argument List:', str(sys.argv))
    else: 
        raise ValueError('Insufficient number of arguments.')
        
    Btar = sys.argv[1]
    R = sys.argv[5]
    
    if Btar == str(90):
        
        B_tar = 25
        delta_nu = 90 / float(R)
        print('delta_nu is: ')
        print(delta_nu)
        
        # target lines and redshifts
        
        nu_rest_tar_list = [115, 230, 345]
        b_list = [b_10_90, b_21_150, b_32_150]
        z_list = [z_10_90, b_21_150, b_21_150]
        P_shot_list = [I10, I21, I32_26]
        
        # interloper lines at those target frequencies and redshifts
        
        z_in_list = [z_10_90, z_21_90, z_32_90, z_43_90, z_54_90, z_65_90, z_76_90]
        b_in_list = [b_10_90, b_21_90, b_32_90, b_43_90, b_54_90, b_65_90, b_76_90]
        
#         nu_rest_tar_list = [nu_10, nu_21, nu_32, nu_43, nu_54, nu_65, nu_76]
#         b_list = b_in_list
#         z_list = z_in_list
        
    elif Btar == str(150):
        
        B_tar = 50
        delta_nu = 150 / float(R) 
        print('delta_nu is: ')
        print(delta_nu)
        
        # target lines and redshifts
    
        nu_rest_tar_list = [230, 345]
    
        b_list = [b_21_150, b_32_150]
        z_list = [b_21_150, b_32_150]
        P_shot_list = [I21, I32_13]
        
        # interloper lines at those target frequencies and redshifts
        
        z_in_list = [z_21_150, z_21_150, z_43_150, z_54_150, z_65_150, z_76_150, z_98_150]
        b_in_list = [b_21_150, b_32_150, b_43_150, b_54_150, b_65_150, b_76_150, b_98_150]
        
#         nu_rest_tar_list = [nu_21, nu_32, nu_43, nu_54, nu_65, nu_76, nu_98]
#         b_list = b_in_list
#         z_list = z_in_list
        
    elif Btar == str(245):
        
        B_tar = 100
        delta_nu = 245  / float(R) 
        print('delta_nu is: ')
        print(delta_nu)
        # target lines and redshifts

        z_list = [z_32_245]
        b_list = [b_32_245]
        P_shot_list = [I32_04]
        
# #         z_in_list = [z_32_245, z_43_245, z_54_245, z_65_245, z_76_245, z_98_245]
# #         b_in_list = [b_32_245, b_43_245, b_54_245, b_65_245, b_76_245, b_98_245]
        
        nu_rest_tar_list = [345]
        
        # interloper lines at those target frequencies and redshifts
        
        z_in_list = [z_32_245, z_43_245, z_54_245, z_65_245, z_76_245, z_98_245]
        b_in_list = [b_32_245, b_43_245, b_54_245, b_65_245, b_76_245, b_98_245]
        
#         nu_rest_tar_list = [nu_32, nu_43, nu_54, nu_65, nu_76, nu_98]
#         b_list = b_in_list
#         z_list = z_in_list
    
    else:
        raise ValueError('Btar is the center of the band and must be 90, 150, or 245 GHz.')
    
    print(sys.argv[1])
    
    print(B_tar)
    
    noise_model = sys.argv[2]
    
    model_selection = sys.argv[3]
    
    first_half = sys.argv[6]
    
    # read in additional parameters - some are defaulted. 
    
    field_length = sys.argv[4] #128 # deg # sky fraction from Chili, 40% --> sqrt(0.4*4 pi Sr * (180/pi)**2) --> Sr --> deg
    # field_length = 45
    # delta_nu = sys.argv[5] #500 Mhz bins

    alpha = -2.8 #-5/3
    Beta = -5/3
    l_knee = 200 #70 #72.9
    eta_min = 0.05
    
    print(field_length)
    
    fisher_list = []
    
    P_n_list = np.logspace(-6, 6, 50)
    
    if first_half == 'first':
        P_n_list_slice = P_n_list[:25]
        filename = "final_npys/fisher_matrix_{}_{}_{}_{}_{}_{}_first_half.npy".format(B_tar, noise_model, model_selection, field_length, R, l_knee)
    elif first_half == 'second': 
        P_n_list_slice = P_n_list[25:]
        filename = "final_npys/fisher_matrix_{}_{}_{}_{}_{}_{}_second_half.npy".format(B_tar, noise_model, model_selection, field_length, R, l_knee)

#     P_n_list_slice = P_n_list[50:]
    

            
    
    elif noise_model == 'RSD_mono':
        print('RSD_mono selected')
        for Pn in P_n_list_slice:
            fisher_matrix = np.array([[0,0], [0,0]])
            for i,nu_rest_tar in enumerate(nu_rest_tar_list):

                fisher_new =  run_forecast_mono(Pn, P_shot_list[i], B_tar, z_list[i], nu_rest_tar,
                b_in_list[:i]+b_in_list[i+1:], z_in_list[:i]+z_in_list[i+1:], b_list[i], delta_nu, '150', 'null', model_selection, float(field_length))
            
                fisher_matrix = fisher_matrix + fisher_new
        
            fisher_list.append(fisher_matrix)
            
    elif noise_model == 'RSD_quad':
        print('RSD_quad selected')
        for Pn in P_n_list_slice:
            fisher_matrix = np.array([[0,0], [0,0]])
            for i,nu_rest_tar in enumerate(nu_rest_tar_list):

                fisher_new =  run_forecast_quad(Pn, P_shot_list[i], B_tar, z_list[i], nu_rest_tar,
                b_in_list[:i]+b_in_list[i+1:], z_in_list[:i]+z_in_list[i+1:], b_list[i], delta_nu, '150', 'null', model_selection, float(field_length))
            
                fisher_matrix = fisher_matrix + fisher_new
        
            fisher_list.append(fisher_matrix)
            
    elif noise_model == 'RSD_mono_interloper':
        print('RSD_mono_interloper selected')
        for Pn in P_n_list_slice:
            fisher_matrix = np.array([[0,0], [0,0]])
            for i,nu_rest_tar in enumerate(nu_rest_tar_list):

                fisher_new =  run_forecast_mono_interloper(Pn, P_shot_list[i], B_tar, z_list[i], nu_rest_tar,
                b_in_list[:i]+b_in_list[i+1:], z_in_list[:i]+z_in_list[i+1:], b_list[i], delta_nu, '150', 'null', model_selection, float(field_length))
            
                fisher_matrix = fisher_matrix + fisher_new
        
            fisher_list.append(fisher_matrix)
            
    elif noise_model == 'RSD_quad_interloper':
        print('RSD_quad_interloper selected')
        for Pn in P_n_list_slice:
            fisher_matrix = np.array([[0,0], [0,0]])
            for i,nu_rest_tar in enumerate(nu_rest_tar_list):

                fisher_new =  run_forecast_quad_interloper(Pn, P_shot_list[i], B_tar, z_list[i], nu_rest_tar,
                b_in_list[:i]+b_in_list[i+1:], z_in_list[:i]+z_in_list[i+1:], b_list[i], delta_nu, '150', 'null', model_selection, float(field_length))
            
                fisher_matrix = fisher_matrix + fisher_new
        
            fisher_list.append(fisher_matrix)
    
            
    elif noise_model == 'fiducial_mitigated':
        print('fiducial mitigated selected')
        for Pn in P_n_list_slice:
            fisher_matrix = np.array([[0,0], [0,0]])
            for i,nu_rest_tar in enumerate(nu_rest_tar_list):

                fisher_new =  run_forecast_fiducial_mitigated(Pn, P_shot_list[i], B_tar, z_list[i], nu_rest_tar,
                b_in_list[:i]+b_in_list[i+1:], z_in_list[:i]+z_in_list[i+1:], b_list[i], delta_nu, '150', 'null', model_selection, float(field_length))
            
                fisher_matrix = fisher_matrix + fisher_new
        
            fisher_list.append(fisher_matrix)
            
    elif noise_model == 'fiducial_unmitigated':
        print('fiducial unmitigated selected')
        for Pn in P_n_list_slice:
            fisher_matrix = np.array([[0,0], [0,0]])
            for i,nu_rest_tar in enumerate(nu_rest_tar_list):

                fisher_new =  run_forecast_fiducial_unmitigated(Pn, P_shot_list[i], B_tar, z_list[i], nu_rest_tar,
                b_in_list[:i]+b_in_list[i+1:], z_in_list[:i]+z_in_list[i+1:], b_list[i], delta_nu, '150', 'null', model_selection, float(field_length))
            
                fisher_matrix = fisher_matrix + fisher_new
        
            fisher_list.append(fisher_matrix)
    
    elif noise_model == 'covariance':
        print('covariance selected')
        for Pn in P_n_list_slice:
            fisher_matrix = np.array([[0,0], [0,0]])
            for i,nu_rest_tar in enumerate(nu_rest_tar_list):

                fisher_new =  run_forecast_covariance(Pn, P_shot_list[i], B_tar, z_list[i], nu_rest_tar,
                b_in_list[:i]+b_in_list[i+1:], z_in_list[:i]+z_in_list[i+1:], b_list[i], delta_nu, '150', 'null', model_selection, float(field_length))
            
                fisher_matrix = fisher_matrix + fisher_new
        
            fisher_list.append(fisher_matrix)
            
    elif noise_model == 'covariance_full':
        print('full covariance selected w/interlopers')
        for Pn in P_n_list_slice:
            fisher_matrix = np.array([[0,0], [0,0]])
            for i,nu_rest_tar in enumerate(nu_rest_tar_list):

                fisher_new =  run_forecast_covariance_full(Pn, P_shot_list[i], B_tar, z_list[i], nu_rest_tar,
                b_in_list[:i]+b_in_list[i+1:], z_in_list[:i]+z_in_list[i+1:], b_list[i], delta_nu, '150', 'null', model_selection, float(field_length))
            
                fisher_matrix = fisher_matrix + fisher_new
        
            fisher_list.append(fisher_matrix)
            
    else:
        raise ValueError('Not a Noise Model')
        
    np.save(filename, fisher_list)
    
    pass

if __name__ == "__main__":
    print('code is running')
    main()


# In[8]:


#print(np.load('fisher_list_DE_test.npy'))


# In[ ]:




