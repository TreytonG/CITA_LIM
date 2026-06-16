import numpy as np
import os
import pandas as pd




"This defines the functional form of the scatter in logL at fixed M_h that I am going to use to reproduce the figures in Liu et al. 2024"
def sigma_logL(M_h, z=6.0,
               sigma_high=1.06, sigma_low=0.20,
               M_pivot=3.7e9, alpha=0.675,
               burstiness=1.0):
    base = sigma_low + burstiness*((sigma_high - sigma_low) * np.exp(-(M_h / M_pivot)**alpha))
    return np.maximum(base, sigma_low)

def sigma_model(M_h, epsilon, epsilon0=0.015,
                sigma_low=0.2, sigma_high=1.13,
                M_pivot=8e9, alpha=0.75, p=2.3):

    # mass dependence (fixed)
    exp_factor = np.exp(-(M_h / M_pivot)**alpha)

    # smooth ε activation
    eff_factor = (epsilon / epsilon0)**p / (1 + (epsilon / epsilon0)**p)

    return sigma_low + (sigma_high - sigma_low) * eff_factor * exp_factor

'''
Pipeline: Read in halo catalog -> assign luminosities with scatter -> compute luminosity function and compare to observations
'''
