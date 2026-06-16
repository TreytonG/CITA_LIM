import numpy as np
import matplotlib.pyplot as plt
from bursty_scatter import sigma_logL_with_epsilon
import scipy.optimize as sci

# =====================================================
# 1. CREATE HALO MASS ARRAY
# =====================================================
# Masses where we evaluate the scatter model
M_h = np.logspace(9, 13, 200)       # 200 mass points, smooth curve
epsilon = 0.005                     # Choose efficiency to test

# =====================================================
# 2. DEFINE MASS–LUMINOSITY RELATION (base model)
# =====================================================
# Placeholder mean luminosity relation:
N, beta = 1e10, 1.5
L_mean = N * (M_h / 1e10)**beta

# =====================================================
# 3. GENERATE MANY GALAXIES PER MASS
# =====================================================
num_per_mass = 200                  # this is new! helps to maybe measure scatter better
M_grid = np.repeat(M_h, num_per_mass)
L_mean_grid = np.repeat(L_mean, num_per_mass)

# Compute model scatter at each repeated mass
sigma_dex = sigma_logL_with_epsilon(M_grid, epsilon=epsilon)
sigma_ln = sigma_dex * np.log(10)   # convert dex → ln

# Draw luminosities with log-normal scatter
L_scattered = L_mean_grid * np.exp(
    np.random.normal(0, sigma_ln)
)

# =====================================================
# 4. MEASURE SCATTER IN LOG LUMINOSITY vs MASS
# =====================================================
bins = np.logspace(9, 13, 25)
bin_centers = 0.5 * (bins[:-1] + bins[1:])
measured_scatter = []

for i in range(len(bins) - 1):
    in_bin = (M_grid >= bins[i]) & (M_grid < bins[i + 1])
    if np.sum(in_bin) > 10:
        logL_vals = np.log10(L_scattered[in_bin])
        measured_scatter.append(np.std(logL_vals))
    else:
        measured_scatter.append(np.nan)

# =====================================================
# 5. PLOT: Measured vs Model True Scatter
# =====================================================
plt.figure(figsize=(7, 5))

# Measured scatter from simulation
plt.plot(bin_centers, measured_scatter, 'o', label='measured (simulation)')

# Model scatter σ(M, ε) (smooth curve)
sigma_true = sigma_logL_with_epsilon(M_h, epsilon=epsilon)
plt.plot(M_h, sigma_true, '-', label='model scatter')

plt.xscale('log')
plt.xlabel('Mh / Msun')
plt.ylabel('σ_logL (dex)')
plt.ylim(0, 1)
plt.title(f'Bursty scatter vs halo mass  (ε = {epsilon})')
plt.legend()
plt.grid(True)
plt.show()