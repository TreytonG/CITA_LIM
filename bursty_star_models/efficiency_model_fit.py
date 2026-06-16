import numpy as np
import pandas as pd
from scipy.optimize import least_squares

# =====================================================
# 1. FILE PATHS
# =====================================================

path_005_logmh   = r"/home/treyton/CITA_LIM/bursty_star_models/epsilon_0p005_log_LCII.csv"
path_005_scatter = r"/home/treyton/CITA_LIM/bursty_star_models/epsilon_0p005_scatter.csv"

path_015_logmh   = r"/home/treyton/CITA_LIM/bursty_star_models/epsilon_0p015_log_mh.csv"
path_015_scatter = r"/home/treyton/CITA_LIM/bursty_star_models/epsilon_0p015_scatter.csv"

path_045_logmh   = r"/home/treyton/CITA_LIM/bursty_star_models/epsilon_0p045_log_mh.csv"
path_045_scatter = r"/home/treyton/CITA_LIM/bursty_star_models/epsilon_0p045_scatter.csv"


# =====================================================
# 2. LOAD DATA
# =====================================================

def load_csv_skip_index(path):
    return pd.read_csv(path, usecols=[1], header=None).values.flatten()


logM_005 = load_csv_skip_index(path_005_logmh)
sig_005  = load_csv_skip_index(path_005_scatter)
eps_005  = np.full_like(logM_005, 0.005)

logM_015 = load_csv_skip_index(path_015_logmh)
sig_015  = load_csv_skip_index(path_015_scatter)
eps_015  = np.full_like(logM_015, 0.015)

logM_045 = load_csv_skip_index(path_045_logmh)
sig_045  = load_csv_skip_index(path_045_scatter)
eps_045  = np.full_like(logM_045, 0.045)


# =====================================================
# 3. COMBINE DATA
# =====================================================

logM_all = np.concatenate([logM_005, logM_015, logM_045])
M_all    = 10 ** logM_all

sig_all  = np.concatenate([sig_005, sig_015, sig_045])
eps_all  = np.concatenate([eps_005, eps_015, eps_045])

xdata = np.vstack([M_all, eps_all])


# =====================================================
# 4. SIMPLIFIED BURSTINESS MODEL
# =====================================================

def sigma_model(x, sigma_low, A, M_pivot, alpha, p):
    """
    Bursty scatter model for LIM luminosity–halo mass relation.

    This model assumes that the total scatter in log-luminosity
    comes from two components:

    1. A baseline (non-bursty) floor: sigma_low
    2. A burst-driven contribution controlled by:
        - halo mass dependence
        - star formation efficiency ε dependence

    ---------------------------------------------------------
    PARAMETERS:

    sigma_low :
        Baseline scatter (dex) in log(L) at fixed halo mass.
        Represents non-bursty / equilibrium star formation noise.
        This is the irreducible floor of the model.

    A :
        Burst amplitude (dex).
        This is the *maximum additional scatter* contributed by
        bursty star formation when:
            - ε is large (f(ε) → 1)
            - halo mass is well below M_pivot

        Physically:
            controls how strong burst-driven stochasticity can get.

    M_pivot :
        Characteristic halo mass scale (in 1e10 Msun units after normalization).
        Sets where burstiness is suppressed at high halo mass.

        Physically:
            above this mass, feedback / regulation reduces burst effects.

    alpha :
        Controls how sharply burstiness is suppressed with halo mass.

        Larger alpha → sharper transition between bursty and non-bursty regimes.

    p :
        Controls sensitivity to star formation efficiency ε.

        Larger p → burstiness turns on more sharply as ε increases.

    ---------------------------------------------------------
    MODEL STRUCTURE:

    σ(M, ε) = σ_low + A × f(ε) × exp[-(M / M_pivot)^α]

    where:

    f(ε) = (ε/ε0)^p / (1 + (ε/ε0)^p)

    ---------------------------------------------------------
    """
    M, eps = x
    eps0 = 0.015  # reference efficiency scale

    # -------------------------------------------------
    # Mass scaling:
    # Low mass halos → more burstiness
    # High mass halos → suppressed scatter
    # -------------------------------------------------
    M_norm = M / 1e10
    M_pivot_norm = M_pivot / 1e10

    mass_term = np.exp(-(M_norm / M_pivot_norm)**alpha)

    # -------------------------------------------------
    # ε (burstiness control):
    # Acts ONLY as an amplitude switch.
    # Ensures:
    #   ε → 0  => no burstiness
    #   ε large => full burst regime
    # -------------------------------------------------
    eps_term = (eps / eps0)**p / (1 + (eps / eps0)**p)

    return sigma_low + A * eps_term * mass_term


# =====================================================
# 5. RESIDUAL FUNCTION
# =====================================================

def residuals(params, x, y):
    return sigma_model(x, *params) - y


# =====================================================
# 6. FIT
# =====================================================

initial_guess = [
    0.2,   # sigma_low : baseline scatter floor
    0.8,   # A         : burst-driven scatter amplitude
    0.5,   # M_pivot   : mass scale of suppression (in 1e10 Msun)
    0.8,   # alpha     : steepness of mass suppression
    2.0    # p         : ε sensitivity (burstiness turn-on)
]

lower_bounds = [0.05, 0.0, 0.1, 0.1, 0.1]
upper_bounds = [2.0,  5.0, 10.0, 5.0, 5.0]

result = least_squares(
    residuals,
    x0=initial_guess,
    bounds=(lower_bounds, upper_bounds),
    args=(xdata, sig_all),
    max_nfev=50000
)

params = result.x


# =====================================================
# 7. OUTPUT
# =====================================================

print("\n=== BURSTINESS FIT ===")
print("sigma_low =", params[0], "  (baseline scatter floor)")
print("A         =", params[1], "  (maximum burst-driven scatter amplitude)")
print("M_pivot   =", params[2], "  (mass scale of burst suppression)")
print("alpha     =", params[3], "  (sharpness of mass dependence)")
print("p         =", params[4], "  (ε sensitivity / burstiness trigger strength)")

print("\nSuccess:", result.success)
print(result.message)
print("Final cost:", result.cost)