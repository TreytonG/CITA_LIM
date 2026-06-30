'''
                                                                 CubeGen
             This script will generate intensity cubes of an input catalog of lightcones, using CITA_LIM tool and dependencies
                            Generated figures will be stored in the output directory specified by the user
This script closesly mirrors the llm_readme file created by Patrick Horlaville, modified to support catalogs rather than single lightcones
'''


# Setting the Directory #
# We want to be working in the CITA_LIM directory at all times
import os
os.chdir("/home/treyton/CITA_LIM")

import warnings

warnings.filterwarnings("ignore", module="hmf.*")


# Importing Neccesary Libraries #
# Base imports
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

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
from zoneinfo import ZoneInfo
from pathlib import Path
import h5py
import glob
import limlam
from scipy.stats import skew, kurtosis

# Defualt plotting settings
matplotlib.rcParams.update({'font.size': 18, 'figure.figsize': [8, 7]})


# Assembling the Lightcone Filepaths #
# Reads off user inputted lightcone paths in a text file, and assembles the full filepaths to be used
# The text file should have one filename per line, and will be appended to the base path "/mnt/AccessArk/lightcone_factory/"
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

# Defualt plotting settings
matplotlib.rcParams.update({'font.size': 18, 'figure.figsize': [8, 7]})


# Get the date and time to timestamp the output files with
now_et = datetime.now(ZoneInfo("America/New_York"))
date = now_et.strftime("%m-%d-%Y")
time = now_et.strftime("%I:%M:%S %p")


# Initialize our LineModel #
# We will use the model "Lichen_v4" created by Patrick Horlaville
m = lim()
m_cii = lim('Lichen_v4_bursty', doSim = True)


# Creating the data storage
eps_values = [0.005, 0.015, 0.045, 0.1]
root = Path("CubeGen Output; " + date + ", " + time)
folder_registry = {}
for eps in eps_values:
    base = root / f"eps={eps}"
    folder_registry[eps] = {"base": base,
                            "map": base / f"{eps}_map", 
                            "beamed_map": base / f"{eps}_beamed_map",  
                            "raw_data": base / f"{eps}_raw_data",
                            "power_spectra": base / f"{eps}_power_spectra",
                            "histogram": base / f"{eps}_histogram"   
                            # "pdf": base / f"{eps}_pdf",
                            }
other_base = root / "Other Figures"
folder_registry["other"] = {"base": other_base, 
                            # "four_panel_maps": other_base / "4_panel_maps", 
                            # "four_panel_beamed_maps": other_base / "4_panel_beamed_maps",   
                            "summary_statistics": other_base / "summary_statistics"
                            }
for eps_dict in folder_registry.values():
    for path in eps_dict.values():
        path.mkdir(parents=True, exist_ok=True)


# Processing the Lightcones #
# The for loop iterates through each lightcone file path, processing and creating figures for each one
count = 0
for i in range(len(filepaths)):
    current_file = filepaths[i]
    print(f"Processing file {i+1}/{len(filepaths)}: {current_file}")


    # Setting up the Parameters of m_cii_0005 #
    # More information on what the parameters do can be found in llm_readme 
    m_cii.update(model_par = {'zdex': 0.4,
    'M0': 1900000000.0,
    'Mmin': 20000000000,
    'M_pivot':1e12,
    'alpha_MH1': 0.74,
    'alpha_LCII': 0.024,
    'alpha0': -1.412,
    'gamma0': 0.31,
    'epsilon': 0.005,
    'BehrooziFile': 'sfr_reinterp.dat'},
                dnu = 2.8*u.GHz,
                nuObs = 270*u.GHz,
                Delta_nu = 40*u.GHz,
                tobs = 40000*u.h,
                Omega_field = 2.25*u.deg**2,
                beam_FWHM = 48*u.arcsec,
                catalogue_file = current_file)
    print("Setting epsilon to 0.005...")

    # Loading in the Intesities #
    print("compiling...")
    cii_cube_0005 = m_cii.maps
    print("compiled!")

    # Setting up the Parameters of m_cii_0015 #
    m_cii.update(model_par = {'zdex': 0.4,
    'M0': 1900000000.0,
    'Mmin': 20000000000,
    'M_pivot':1e12,
    'alpha_MH1': 0.74,
    'alpha_LCII': 0.024,
    'alpha0': -1.412,
    'gamma0': 0.31,
    'epsilon': 0.015,
    'BehrooziFile': 'sfr_reinterp.dat'},
                dnu = 2.8*u.GHz,
                nuObs = 270*u.GHz,
                Delta_nu = 40*u.GHz,
                tobs = 40000*u.h,
                Omega_field = 2.25*u.deg**2,
                beam_FWHM = 48*u.arcsec,
                catalogue_file = current_file)
    print("Setting epsilon to 0.015...")
    print("compiling...")
    cii_cube_0015 = m_cii.maps
    print("compiled!")

    # Setting up the Parameters of m_cii_0045 #
    m_cii.update(model_par = {'zdex': 0.4,
    'M0': 1900000000.0,
    'Mmin': 20000000000,
    'M_pivot':1e12,
    'alpha_MH1': 0.74,
    'alpha_LCII': 0.024,
    'alpha0': -1.412,
    'gamma0': 0.31,
    'epsilon': 0.045,
    'BehrooziFile': 'sfr_reinterp.dat'},
                dnu = 2.8*u.GHz,
                nuObs = 270*u.GHz,
                Delta_nu = 40*u.GHz,
                tobs = 40000*u.h,
                Omega_field = 2.25*u.deg**2,
                beam_FWHM = 48*u.arcsec,
                catalogue_file = current_file)
    print("Setting epsilon to 0.045...")
    print("compiling...")
    cii_cube_0045 = m_cii.maps
    print("compiled!")

    # Setting up the Parameters of m_cii_01 #
    m_cii.update(model_par = {'zdex': 0.4,
    'M0': 1900000000.0,
    'Mmin': 20000000000,
    'M_pivot':1e12,
    'alpha_MH1': 0.74,
    'alpha_LCII': 0.024,
    'alpha0': -1.412,
    'gamma0': 0.31,
    'epsilon': 0.1,
    'BehrooziFile': 'sfr_reinterp.dat'},
                dnu = 2.8*u.GHz,
                nuObs = 270*u.GHz,
                Delta_nu = 40*u.GHz,
                tobs = 40000*u.h,
                Omega_field = 2.25*u.deg**2,
                beam_FWHM = 48*u.arcsec,
                catalogue_file = current_file)
    print("Setting epsilon to 0.1...")
    print("compiling...")
    cii_cube_01 = m_cii.maps
    print("compiled!")

    # Create a dictionary of the 4 maps
    eps_cubes = {0.005: cii_cube_0005,
                 0.015: cii_cube_0015,
                 0.045: cii_cube_0045,
                 0.100: cii_cube_01,}

    # Figure Generation #
    # [CII] Noiseless Mock
    for eps, current_map in eps_cubes.items():
        fig, axes = plt.subplots(nrows = 1, ncols = 1, figsize = (10, 8))
        plt.subplot(111)
        plt.imshow(current_map[:, :, 6].value, 
                   vmax = 1600, cmap =  'viridis', 
                   extent = [-1, 1, -1, 1], rasterized = True)
        plt.title('[C II] Noiseless Mock at z $\\approx$ 8')
        plt.ylabel('$\\Delta$Dec (deg)')
        plt.xlabel('$\\Delta$RA (deg)')
        plt.colorbar(label = '$I_{\\rm [C\\, II]}$ (Jy sr$^{-1}$)', pad = 0.03)
        
        # Save the figure
        save_dir = folder_registry[eps]["map"]
        filename = f"Map_{count:03d}_eps{eps}.png"
        save_path = save_dir / filename
        fig.savefig(save_path, dpi=300)
        print(f"Saved {filename} into {save_path}")
        plt.close(fig)

    # [CII] Beamed Noiseless Mock
    for eps, current_map in eps_cubes.items():
        fig, axes = plt.subplots(nrows = 1, ncols = 1, figsize = (10, 8))
        plt.subplot(111)
        plt.imshow(gaussian_filter(current_map[:, :, 6], 1), 
                   vmax = 1600, cmap =  'viridis', 
                   extent = [-1, 1, -1, 1], rasterized = True)
        plt.title('[C II] Beamed Noiseless Mock at z $\\approx$ 8')
        plt.ylabel('$\\Delta$Dec (deg)')
        plt.xlabel('$\\Delta$RA (deg)')
        plt.colorbar(label = '$I_{\\rm [C\\, II]}$ (Jy sr$^{-1}$)', pad = 0.03)

        # Save the figure
        save_dir = folder_registry[eps]["beamed_map"]
        filename = f"Beamed_{count:03d}_eps{eps}.png"
        save_path = save_dir / filename
        fig.savefig(save_path, dpi=300)
        print(f"Saved {filename} into {save_path}")
        plt.close(fig)

    # Raw Data Storage
    for eps, current_map in eps_cubes.items():
        save_dir = folder_registry[eps]["raw_data"]
        filename = f"RawData_{count:03d}_eps{eps}.h5"
        save_path = save_dir / filename
        with h5py.File(save_path, "w") as f:
            f["intensity"] = current_map
        print(f"Saved {filename} into {save_path}")
    print(f"Finished with lightcone {count:03d}")

    # Intensity Histogram
    for eps, current_map in eps_cubes.items():
        fig, axes = plt.subplots(nrows = 1, ncols = 1, figsize = (10, 8))
        vals = current_map.flatten().value
        vals = vals[vals > 0]
        plt.hist(vals, bins=np.logspace(np.log10(vals.min()),   
                                        np.log10(vals.max()), 50))

        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Intensity (Jy/sr)')
        plt.ylabel('Count')

        # Save the figure
        save_dir = folder_registry[eps]["histogram"]
        filename = f"Histogram{count:03d}_eps{eps}.png"
        save_path = save_dir / filename
        fig.savefig(save_path, dpi=300)
        print(f"Saved {filename} into {save_path}")
        plt.close(fig)

    # Summary Statistics
    rows = []
    zidx = 6
    for eps, cube in eps_cubes.items():
        data = cube[:, :, zidx]
        rows.append({'epsilon': eps, 'mean': np.mean(data), 'std': np.std(data),   
                    'variance': np.var(data), 'skewness': skew(data.flatten()),    
                    'kurtosis': kurtosis(data.flatten()),  
                    'p99': np.percentile(data,99), 
                    'p999': np.percentile(data,99.9),  
                    'max': np.max(data)})
    stats_df = pd.DataFrame(rows)

    # Save the figure
    save_dir = folder_registry["other"]["summary_statistics"]
    filename = f"summary_statistics_{count:03d}.csv"
    save_path = save_dir / filename
    stats_df.to_csv(save_path, index=False)
    print(f"Saved {filename} into {save_path}")

    # Power Spectra
    for eps, current_map in eps_cubes.items():
        m = lim("Lichen_v4_bursty", doSim=True)
        m.update(model_par = {'zdex': 0.4, 'M0': 1900000000.0,
                              'Mmin': 20000000000, 'M_pivot':1e12,
                              'alpha_MH1': 0.74, 'alpha_LCII': 0.024,
                              'alpha0': -1.412, 'gamma0': 0.31,
                              'epsilon': eps,'BehrooziFile': 'sfr_reinterp.dat'},
                                dnu = 2.8*u.GHz,
                                nuObs = 270*u.GHz,
                                Delta_nu = 40*u.GHz,
                                tobs = 40000*u.h,
                                Omega_field = 2.25*u.deg**2,
                                beam_FWHM = 48*u.arcsec)
        m.limlam_cosmo.Omega_L = 0.691
        m.limlam_cosmo.Omega_M = 0.307
        m.limlam_cosmo.Omega_B = 0.048
        m.limlam_cosmo.h = 0.677
        m.limlam_cosmo.ns = 0.96

        m.mapinst.maps = current_map
        pk_dict = limlam.llm.map_to_pspec(m.mapinst, m.limlam_cosmo)
        k = pk_dict['k']
        Pk = pk_dict['Pk']

        plt.figure(figsize=(6,5))
        plt.plot(k, Pk, lw=2)
        plt.xlabel(r'$k$')
        plt.ylabel(r'$P(k)$')
        plt.title('Power Spectrum')
        plt.grid(alpha=0.3)
        plt.loglog()

        # Save the figure
        save_dir = folder_registry[eps]["power_spectra"]
        filename = f"Power_Spectra_{count:03d}_eps{eps}.png"
        save_path = save_dir / filename
        fig.savefig(save_path, dpi=300)
        print(f"Saved {filename} into {save_path}")
        plt.close(fig)

    count += 1