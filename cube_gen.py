'''
                                                              CubeGen Script
             This script will generate intensity cubes of an input catalog of lightcones, using CITA_LIM tool and dependencies
                            Generated figures will be stored in the output directory specified by the user
This script closesly mirrors the llm_readme file created by Patrick Horlaville, modified to support catalogs rather than single lightcones
'''


# Setting the Directory #
# We want to be working in the CITA_LIM directory at all times
import os
os.chdir("/home/treyton/CITA_LIM")

import warnings

warnings.filterwarnings(
    "ignore",
    module="hmf.*"
)

# Assembling the Lightcone Filepaths #
# Reads off user inputted lightcone paths in a text file, and assembles the full filepaths to be used
# The text file should have one filename per line, and will be appended to the base path "/mnt/AccessArk/lightcone_factory/"
import os
def get_filepaths(textfile="/home/treyton/CITA_LIM/cube_gen_test_lightcone_catalog", base_path="/mnt/AccessArk/lightcone_factory/", verbose=True):
    filepaths = []
    invalid_paths = []

    with open(textfile, 'r') as f:
        for line in f:
            p = line.strip()
            if not p:
                continue
            full_path = os.path.join(base_path, p)

            # Check existence and readability
            if os.path.isfile(full_path) and os.access(full_path, os.R_OK):
                filepaths.append(full_path)
            else:
                invalid_paths.append(full_path)
    # Prints a warning if some files do not exist or unreadable, but continues with the valid files
    if verbose and invalid_paths:
        print(f"Warning: {len(invalid_paths)} invalid/unreadable files found.")
        for p in invalid_paths[:10]:  # show first few only
            print("  Missing/unreadable:", p)

    return filepaths
filepaths = get_filepaths()



# Importing Neccesary Libraries #
# Base imports
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# Math packages
import math
from scipy import special
from scipy import interpolate
import astropy.units as u
from scipy.ndimage import gaussian_filter

# base intensity mapping package
from lim import lim
from limlam_mocker.limlam_mocker import empty_table

# other miscellaneous packages
from datetime import datetime

# Defualt plotting settings
matplotlib.rcParams.update({'font.size': 18, 'figure.figsize': [8, 7]})



# Initialize our LineModel #
# We will use the model "Lichen_v4" created by Patrick Horlaville
m = lim()
m_cii = lim('Lichen_v4', doSim = True)



# Processing the Lightcones #
# The for loop iterates through each lightcone file path, processing and creating figures for each one
for i in range(len(filepaths)):
    current_file = filepaths[i]
    print(f"Processing file {i+1}/{len(filepaths)}: {current_file}")


    # Setting up the Parameters of m_cii #
    # More information on what the parameters do can be found in llm_readme 
    m_cii.update(model_par = {'zdex': 0.4,
        'M0': 1900000000.0,
        'Mmin': 20000000000,
        'alpha_MH1': 0.74,
        'alpha_LCII': 0.024,
        'alpha0': -1.412,
        'gamma0': 0.31,
        'BehrooziFile': 'sfr_reinterp.dat'},
            dnu = 2.8*u.GHz,
            nuObs = 270*u.GHz,
            Delta_nu = 40*u.GHz,
            tobs = 40000*u.h,
            Omega_field = 2.25*u.deg**2,
            beam_FWHM = 48*u.arcsec,
            catalogue_file = current_file)
    print("updated!")
    


    # Loading in the Intesities #
    # The following line is the bulk of the compute time, as it loads in all of the data from the lightcone
    cii_cube = m_cii.maps
    print("compiled")

     # Figure Generation #
    # TODO: Think about which figures would be good to have and store
    # [CII] Noiseless Mock
    fig, axes = plt.subplots(nrows = 1, ncols = 1, figsize = (10, 8))
    plt.subplot(111)
    plt.imshow(cii_cube[:, :, 6].value, vmax = 2000, cmap =  'viridis', extent = [-1, 1, -1, 1], rasterized = True)
    plt.title('[C II] Noiseless Mock at z $\\approx$ 6')
    plt.ylabel('$\\Delta$Dec (deg)')
    plt.xlabel('$\\Delta$RA (deg)')
    plt.colorbar(label = '$I_{\\rm [C\\, II]}$ (Jy sr$^{-1}$)', pad = 0.03)
    # TODO: save the figure output to a sensible file

    # [CII] Beamed Noiseless Mock
    fig, axes = plt.subplots(nrows = 1, ncols = 1, figsize = (10, 8))
    plt.subplot(111)
    plt.imshow(gaussian_filter(cii_cube[:, :, 6], 1), vmax = 2000, cmap =  'viridis', extent = [-1, 1, -1, 1], rasterized = True)
    plt.title('[C II] Beamed Noiseless Mock')
    plt.ylabel('$\\Delta$Dec (deg)')
    plt.xlabel('$\\Delta$RA (deg)')
    plt.colorbar(label = '$I_{\\rm [C\\, II]}$ (Jy sr$^{-1}$)', pad = 0.03)
    # TODO: save the figure output to a sensible file
