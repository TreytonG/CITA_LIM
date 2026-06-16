from __future__ import absolute_import, print_function
import numpy as np
import scipy
from .tools import * # . tools? 
from . import debug
import time
import h5py
from limlam_mocker.limlam_mocker import empty_table

# [TREY]: New filetype support for .h5py filetype
def load_lightcone_catalogue(filein, cosmo):
    halos = empty_table()
    halo_info = filein

    halos.M = np.array(halo_info['m'])

    halos.x_pos = np.array(halo_info['x'])/cosmo.h
    halos.y_pos = np.array(halo_info['y'])/cosmo.h
    halos.z_pos = np.array(halo_info['z'])/cosmo.h

    halos.chi = np.sqrt(halos.x_pos**2+halos.y_pos**2+halos.z_pos**2)

    halos.redshift = np.array(halo_info['zcos'])
    halos.redshifto = np.array(halo_info['zlos'])

    halos.nhalo = len(halos.M)

    halos.ra = np.array(halo_info['ra'])
    halos.dec = np.array(halo_info['dec'])

    halos.sm = np.array(halo_info['sm'])
    halos.sfr = np.array(halo_info['sfr'])
    halos.a_uv = np.array(halo_info['a_uv'])

    return halos


def load_peakpatch_catalogue(halo_info, filetype='.npz', saveHalos=False, saveFolder='/outputs/'):
    """
    Load peak patch halo catalogue into halos class and cosmology into cosmo class
    
    Slightly modified to work with the lim functions, cosmo class split into separate
    function
    
    When save=True, the function will save the Mcen, Msat, cen_pos, sat_pos output to the saveFolder directory.
    
    Default filetype = '.npz'
    Can take filetype = '.h5'

    Returns
    -------
    halos : class
        Contains all halo information (position, redshift, etc..)
    """
    
    # These are the directories from when saveHalos=True. 
    
    Mcen_file    = saveFolder + 'Mcen.dat'    
    Msat_file    = saveFolder + 'Msat.dat'        
    cen_pos_file = saveFolder + 'cen_pos.dat'   
    sat_pos_file = saveFolder + 'sat_pos.dat'        

    halos       = empty_table()                      # creates empty class to put any halo info into 
    
    if filetype=='.h5':
        #[TREY]: Edited functionality of load_peakpatch_catalogue to support .h5py filetype instead of .h5
        from astropy.cosmology import Planck13 as cosmo
        return load_lightcone_catalogue(halo_info, cosmo)
    else:
        print('Loading .npz catalogues...')
        params_dict = halo_info['cosmo_header'][()]     
        if debug.verbose: print("\thalo catalogue contains:\n\t\t", halo_info.files)    #.npz version
            
        cen_x_fov  = params_dict.get('cen_x_fov', 0.) #if the halo catalogue is not centered along the z axis
        cen_y_fov  = params_dict.get('cen_y_fov', 0.) #if the halo catalogue is not centered along the z axis
        
        halos.M          = halo_info['M']     # halo mass in Msun 
        halos.x_pos      = halo_info['x']     # halo x position in comoving Mpc 
        halos.y_pos      = halo_info['y']     # halo y position in comoving Mpc 
        halos.z_pos      = halo_info['z']     # halo z position in comoving Mpc 
        
        halos.vx         = halo_info['vx']    # halo x velocity in km/s
        halos.vy         = halo_info['vy']    # halo y velocity in km/s
        halos.vz         = halo_info['vz']    # halo z velocity in km/s
        halos.redshift   = halo_info['zhalo'] # observed redshift incl velocities
        halos.zformation = halo_info['zform'] # formation redshift of halo
        halos.chi        = np.sqrt(halos.x_pos**2+halos.y_pos**2+halos.z_pos**2)  
        halos.nhalo      = len(halos.M)
        halos.ra         = np.arctan2(-halos.x_pos,halos.z_pos)*180./np.pi - cen_x_fov
        halos.dec        = np.arcsin(  halos.y_pos/halos.chi  )*180./np.pi - cen_y_fov

    assert np.max(halos.M) < 1.e17,             "Halos seem too massive"
    # assert np.max(halos.redshift) < 4.,         "need to change max redshift interpolation in tools.py" 
    assert np.max(halos.redshift) < 10,        "need to change max redshift interpolation in tools.py" # changed for George's z=4.77-6.21 sim data because of error in calling m.maps (May 17) - Clara
    if debug.verbose: print('\n\t%d halos loaded' % halos.nhalo)
    
    return halos


def load_peakpatch_catalogue_cosmo(halo_info):
    """
    Load peak patch cosmology into cosmo class
    
    Slightly modified to work with the lim functions, halo class split into separate
    function

    Returns
    -------
    cosmo : class
        Contains all cosmology information (Omega_i, sigme_8, etc)
    """
    cosmo      = empty_table()            # creates empty class to put any cosmology info into  

    if debug.verbose: print("\thalo catalogue contains:\n\t\t", halo_info.files)
    
    #get cosmology from halo catalogue
    params_dict    = halo_info['cosmo_header'][()]
    cosmo.Omega_M  = params_dict.get('Omega_M')
    cosmo.Omega_B  = params_dict.get('Omega_B')
    cosmo.Omega_L  = params_dict.get('Omega_L')
    cosmo.h        = params_dict.get('h'      )
    cosmo.ns       = params_dict.get('ns'     )
    cosmo.sigma8   = params_dict.get('sigma8' )

    assert (cosmo.Omega_M + cosmo.Omega_L)==1., "Does not seem to be flat universe cosmology" 

    if debug.verbose: print('\n\t%d halos loaded' % halos.nhalo)

    return cosmo
    


#@timeme
def cull_peakpatch_catalogue(halos, min_mass, max_mass, mapinst, haloType='all'): 
    """
    crops the halo catalogue to only include desired halos
    haloType determines whether you want to use all halos (centrals and satellites), only centrals,
    or only satellites. Options are:
    haloType='all' for both
            ='cen' for only centrals
            ='sat' for only satellites
    """
    
    dir_halos = dir(halos)
    del_attr  = []
    
    if haloType == 'all':
        Mass     = halos.M
        Redshift = halos.redshift
        RA       = halos.ra
        DEC      = halos.dec
        halos.nhalo_all = halos.nhalo
        for attr in dir_halos:      # remove only centrals and satellites attributes
            if (('cen' in attr) or ('sat' in attr)):
                del_attr.append(attr)
    elif haloType == 'cen':
        Mass     = halos.Mcen
        Redshift = halos.redshift_cen
        RA       = halos.ra_cen
        DEC      = halos.dec_cen
        for attr in dir_halos:      # remove attributes that are not centrals
            if 'cen' not in attr:
                del_attr.append(attr)
    elif haloType == 'sat':
        Mass     = halos.Msat
        Redshift = halos.redshift_sat
        RA       = halos.ra_sat
        DEC      = halos.dec_sat
        for attr in dir_halos:      # remove attributes that are not satellites
            if 'sat' not in attr:
                del_attr.append(attr)
    else:
        raise Exception("haloType only takes 'all', 'cen', or 'sat'.")
    
    [dir_halos.remove(attr) for attr in del_attr]
    
    dm = (Mass > min_mass)  * (Redshift    >= mapinst.z_i)\
                            * (np.abs(RA)  <= mapinst.fov_x/2)\
                            * (np.abs(DEC) <= mapinst.fov_y/2)\
                            * (Redshift    <= mapinst.z_f)\
                            * (Mass        <  max_mass)
    
    for i in dir_halos:
        if i[0]=='_': continue
        try:
            setattr(halos,i,getattr(halos,i)[dm])
        except TypeError:
            pass
    
    if haloType == 'cen':
        halos.nhalo    = len(halos.Mcen)
        halos.M        = halos.Mcen
        halos.chi      = halos.chi_cen
        halos.ra       = halos.ra_cen
        halos.dec      = halos.dec_cen
        halos.redshift = halos.redshift_cen
        halos.x_pos    = halos.x_pos_cen
        halos.y_pos    = halos.y_pos_cen
        halos.z_pos    = halos.z_pos_cen
    elif haloType == 'sat':
        halos.nhalo    = len(halos.Msat)
        halos.M        = halos.Msat
        halos.chi      = halos.chi_sat
        halos.ra       = halos.ra_sat
        halos.dec      = halos.dec_sat
        halos.redshift = halos.redshift_sat
        halos.x_pos    = halos.x_pos_sat
        halos.y_pos    = halos.y_pos_sat
        halos.z_pos    = halos.z_pos_sat
    else:
        halos.nhalo = len(halos.M)
    
    if debug.verbose: print('\n\t%d halos remain after mass/map cut' % halos.nhalo)

    return halos
