import numpy as np

def load_smhm_params(fname, path=r"/Users/gabenell/Documents/Cornell/Files/umachine-dr1/data/smhm/params/"):
    """
    sort of a test, to see if we can use this to load several sets of parameters to plot things in parallel
    Inputs: file name, path to file containing UM DR-1 SMHM parameters (str)
    (from table in Appendix J of Behroozi et al 2019, https://arxiv.org/pdf/1806.07893)
    Outputs: dictionary with each parameter assigned to its name
    """
    full_path = path+fname
    # keys
    param_names = np.array(["eps_0", "eps_a", "eps_lna", "eps_z", "M_0", "M_a", "M_lna", "M_z", "alpha_0", "alpha_a", "alpha_lna", "alpha_z", "beta_0", "beta_a", "beta_z", "delta_0", "gamma_0", "gamma_a", "gamma_z"])
    # values
    param_set = np.loadtxt(full_path)
    param_dict = {param: values[1] for param, values in zip(param_names, param_set)}
    return param_dict
def hm_maxes(z_min, z_max, z_step, fname=r"no_halos.txt", path=r"/Users/gabenell/Documents/Cornell/Files/"):
    """
    loads file from bitbucket of halo mass maximums as a function of redshift
    (for purpose of cutting off plots when halos get too unrealistic)
    inputs: z plotting parameters (floats), filename (str), path (str)
    outputs: maxes corresponding to each z value displayed in the plot
    """
    full_path = path+fname
    maxes = np.loadtxt(full_path)
    z_values = np.round(np.arange(z_min, z_max+z_step, z_step), decimals=1)
    corr_maxes = np.array([maxes[10 * int(z_val)][1] for z_val in z_values])
    return corr_maxes

def transform_pars(raw_pars):
    """
    inputs: parameters loaded from UM DR-1 (dict)

    outputs: a lambda function taking redshift as input
    which outputs a dictionary of relevant weights for the 
    SMHM relation
    """

    a = lambda z: 1 / (1+z)
    log10_M1 = lambda z: raw_pars["M_0"] + raw_pars["M_a"]*(a(z)-1) - raw_pars["M_lna"]*np.log(a(z)) + raw_pars["M_z"]*z
    eps = lambda z: raw_pars["eps_0"] + raw_pars["eps_a"]*(a(z)-1) - raw_pars["eps_lna"]*np.log(a(z)) + raw_pars["eps_z"]*z
    alpha = lambda z: raw_pars["alpha_0"] + raw_pars["alpha_a"]*(a(z)-1) - raw_pars["alpha_lna"]*np.log(a(z)) + raw_pars["alpha_z"]*z
    beta = lambda z: raw_pars["beta_0"] + raw_pars["beta_a"]*(a(z)-1) + raw_pars["beta_z"]*z
    delta = raw_pars["delta_0"]
    log10_gamma = lambda z: raw_pars["gamma_0"] + raw_pars["gamma_a"]*(a(z)-1) + raw_pars["gamma_z"]*z

    # place all of these in a dictionary for use in relation
    weights = lambda z: {
        "log10_M1": log10_M1(z),
        "eps": eps(z),
        "alpha": alpha(z),
        "beta": beta(z),
        "delta": delta,
        "log10_gamma": log10_gamma(z)
    }
    return weights

def smhm_2019(weights):
    """
    stellar mass to halo mass relation from Behroozi 2019:
    https://arxiv.org/pdf/1806.07893

    Inputs: weights with relevant keys (dict)
    Outputs: a lambda function which takes halo mass as input and outputs stellar mass
    """

    x = lambda log10_M_peak, z: log10_M_peak - weights(z)["log10_M1"]
    log10_M_star = lambda log10_M_peak, z: weights(z)["eps"] - np.log10(10**(-1*weights(z)["alpha"]*x(log10_M_peak, z))+10**(-1*weights(z)["beta"]*x(log10_M_peak, z))) + (10**weights(z)["log10_gamma"])*np.exp(-0.5*(x(log10_M_peak, z)/weights(z)["delta"])**2) + weights(z)["log10_M1"]

    return log10_M_star