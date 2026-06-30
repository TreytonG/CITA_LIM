'''
                                                                 CubeGen
             This script will generate intensity cubes of an input catalog of lightcones, using CITA_LIM tool and dependencies
                            Generated figures will be stored in the output directory specified by the user
This script closesly mirrors the llm_readme file created by Patrick Horlaville, modified to support catalogs rather than single lightcones

TEMP: THIS ADDS POWER SPECTRA< HISTOGRAMS AND STATS TO THE MASTER DIRECTORY AND THIS WILL BE USELESS AFTER THAT
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

# Initialize our LineModel #
# We will use the model "Lichen_v4" created by Patrick Horlaville
m = lim()
m_cii = lim('Lichen_v4_bursty', doSim = True)

eps_values = [0.005, 0.015, 0.045, 0.1]
def get_filepaths(textfile="/home/treyton/CITA_LIM/cube_gen_test_lightcone_catalog", verbose=True):
    filepaths = []
    invalid_paths = []
    with open(textfile, 'r') as f:
        for line in f:
            full_path = line.strip()
            if not full_path:
                continue

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

root = Path("CubeGen Output; 06-22-2026, 11:48:18 PM")
folder_registry = {}

for eps in eps_values:
    base = root / f"eps={eps}"
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


count = 0
eps_count = 0
for i in range(len(filepaths)):
    if count == 16:
        count = 0
        eps_count += 1
    current_file = filepaths[i]
    print(f"Processing file {i+1}/{len(filepaths)}: {current_file}")
    h = h5py.File(current_file, 'r')

    # Histogram
    vals = np.array(h['intensity'])
    vals = vals[vals > 0]

    plt.hist(vals, bins=np.logspace(np.log10(vals.min()), 
                                    np.log10(vals.max()), 50))

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Intensity (Jy/sr)')
    plt.ylabel('Count')

    save_dir = folder_registry[eps_values[eps_count]]["histogram"]
    filename = f"Histogram{count:03d}_eps{eps_values[eps_count]}.png"
    save_path = save_dir / filename
    plt.savefig(save_path, dpi=300)
    print(f"Saved {filename} into {save_path}")
    plt.close()

    # Power Spectra
    h = h5py.File(current_file, 'r')
    i = np.array(h['intensity'])
    m = lim('Lichen_v4_bursty',doSim=True)
    m.update(model_par = {'zdex': 0.4, 'M0': 1900000000.0,  
                          'Mmin': 20000000000, 'M_pivot':1e12,  
                          'alpha_MH1': 0.74, 'alpha_LCII': 0.024,   
                          'alpha0': -1.412, 'gamma0': 0.31,
                          'epsilon': eps_values[eps_count], 
                          'BehrooziFile': 'sfr_reinterp.dat'},
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
    m.mapinst.maps = i
    pk_dict = limlam.llm.map_to_pspec(m.mapinst, m.limlam_cosmo)

    # Extract arrays
    k = pk_dict['k']
    Pk = pk_dict['Pk']

    # Plot
    plt.figure(figsize=(6,5))
    plt.plot(k, Pk, lw=2)

    plt.xlabel(r'$k$')
    plt.ylabel(r'$P(k)$')
    plt.title('Power Spectrum')
    plt.grid(alpha=0.3)
    plt.loglog()

    save_dir = folder_registry[eps_values[eps_count]]["power_spectra"]
    filename = f"Power_Spectra{count:03d}_eps{eps_values[eps_count]}.png"
    save_path = save_dir / filename
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Saved {filename} into {save_path}") 
    plt.close()

    count += 1

# Summary Statistics shenanigans
filepaths = get_filepaths(textfile="/home/treyton/CITA_LIM/stat_file")
count = 0
acc = 0
stat_list = [None] * 4
for i in range(len(filepaths)):
    current_file = filepaths[i]
    if count == 3:
        stat_list[count] = current_file
        count = 0
    else:
        stat_list[count] = current_file
        count += 1
        continue
    rows = []
    zidx = 6
    count2 = 0
    for eps in eps_values:
        h = h5py.File(stat_list[count2], 'r')
        cube = np.array(h['intensity'])
        data = cube[:, :, zidx]
        rows.append({'epsilon': eps, 'mean': np.mean(data), 'std': np.std(data),   
                        'variance': np.var(data), 'skewness': skew(data.flatten()),    
                        'kurtosis': kurtosis(data.flatten()),  
                        'p99': np.percentile(data,99), 
                        'p999': np.percentile(data,99.9),  
                        'max': np.max(data)})
        count2 += 1
    stats_df = pd.DataFrame(rows)
    stats_df

    save_dir = folder_registry["other"]["summary_statistics"]
    filename = f"summary_statistics_{acc:03d}.csv"
    save_path = save_dir / filename
    stats_df.to_csv(save_path, index=False)
    print(f"Saved {filename} into {save_path}")

    acc += 1

print("Figures and Statistics added successfully!")



