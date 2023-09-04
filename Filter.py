#!/bin/env python3

import numpy as np
import jax.numpy as jnp
from jax import vmap, jit
from jax.debug import print as jprint
from scipy.interpolate import interp1d
import os
import pandas as pd
import matplotlib.pyplot as plt
#import sedpy
import munch
from typing import NamedTuple, Any
import pathlib
from collections import namedtuple

# Place to store the filters for use with sedpy
save_dir = os.path.abspath(os.path.join('.', 'EmuLP', 'data', 'filters'))
pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)

lightspeed = 2.998e18  # AA/s
ab_gnu = 3.631e-20  # AB reference spctrum in erg/s/cm^2/Hz

sedpyFilter = namedtuple('sedpyFilter', ['name', 'wavelengths', 'transmission'])

# NUV
_wls = np.arange(1000., 3000., 1.)
nuv_transm = np.zeros_like(_wls)
nuv_transm[(_wls>=2100.)*(_wls<=2500.0)] = 1.0
NUV_filt = sedpyFilter(98, _wls, nuv_transm)

# NIR
_wls = np.arange(20000., 25000., 1.)
nir_transm = np.zeros_like(_wls)
nir_transm[(_wls>=21000.)*(_wls<=23000.0)] = 1.0
NIR_filt = sedpyFilter(99, _wls, nir_transm)

def some_hash_function(x):
    return int(jnp.sum(x))

class HashableArrayWrapper:
    def __init__(self, val):
        self.val = val
    def __hash__(self):
        return some_hash_function(self.val)
    def __eq__(self, other):
        return (isinstance(other, HashableArrayWrapper) and
                jnp.all(jnp.equal(self.val, other.val)))

class Filter(munch.Munch):
    """Class for filters implementation."""
    def __init__(self, ident, filterfile, trans_type, filtdict=None):
        if filtdict is None:
            super().__init__()
            self.id = ident
            #self.path = os.path.abspath(filterfile)
            _wls, _trans = np.loadtxt(os.path.abspath(filterfile), unpack=True)
            self.trans_type = trans_type
            wls, transm = sort(_wls, _trans)
            self.mean_wl, new_trans = transform(wls, transm, self.trans_type)
            self.wavelengths, self.transmit,\
                                self.min_wl, self.max_wl,\
                                self.minWL_peak, self.maxWL_peak, self.peak_wl = clip_filter(wls, new_trans)
        else:
            super().__init__(filtdict)
        
        f_transmit = lambda x : jnp.interp(x, self.wavelengths, self.transmit, left=0., right=0., period=None)
        
        save_dir = os.path.abspath(os.path.join('.', 'EmuLP', 'data', 'filters'))
        import pathlib
        pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)
        np.savetxt(os.path.join(save_dir, f'{self.id}.par'),\
                   np.column_stack((self.wavelengths,\
                                    f_transmit(self.wavelengths)\
                                    )\
                                   )\
                  )
        self.sedpy_filt = sedpy.observate.Filter(f'{self.id}', directory=save_dir)
        
    def __str__(self):
        return f"{self.id} filter" #, from file : {self.path}."
        
    def __repr__(self):
        return f"<Filter object : name={self.id}" # ; path={self.path}>"
    
    def __eq__(self, other):
        return (self.id == other.id) and (self.trans_type == other.trans_type) and (self.mean_wl == other.mean_wl) and (self.minWL_peak == other.minWL_peak) and (self.maxWL_peak == other.maxWL_peak) and (self.peak_wl == other.peak_wl)
    
    def __ne__(self, other):
        return not self.__eq__(other)
    
    def __hash__(self):
        return hash((self.id, self.trans_type, self.mean_wl, self.minWL_peak, self.maxWL_peak, self.peak_wl))
        
def sort(wl, trans):
    _inds = jnp.argsort(wl)
    wls = wl[_inds]
    transm = trans[_inds]
    return wls, transm

def lambda_mean(wls, trans):
    mean_wl = jnp.trapz(wls*trans, wls) / jnp.trapz(trans, wls)
    return mean_wl
        
def clip_filter(wls, trans):
    _indToKeep = np.where(trans >= 0.01*np.max(trans))[0]
    new_wls = wls[ _indToKeep ]
    new_trans = trans[ _indToKeep ]
    _lambdas_peak = new_wls[ np.where(new_trans >= 0.5*np.max(new_trans))[0] ]
    min_wl, max_wl = np.min(new_wls), np.max(new_wls)
    minWL_peak, maxWL_peak = np.min(_lambdas_peak), np.max(_lambdas_peak)
    peak_wl = new_wls[ np.argmax(new_trans) ]
    return new_wls, new_trans, min_wl, max_wl, minWL_peak, maxWL_peak, peak_wl
        
def transform(wls, trans, trans_type):
    mean_wl = lambda_mean(wls, trans)
    if (trans_type.lower() == "photons") and (mean_wl>0.):
        new_trans = jnp.array( trans*wls/mean_wl )
    else:
        new_trans = trans
    return mean_wl, new_trans

def load_filt(ident, filterfile, trans_type):
    __wls, __trans = np.loadtxt(os.path.abspath(filterfile), unpack=True)
    _wls, _trans = jnp.array(__wls), jnp.array(__trans)
    wls, transm = sort(_wls, _trans)
    mean_wl, new_trans = transform(wls, transm, trans_type)
    max_trans = jnp.max(new_trans)
    _sel = (new_trans >= 0.01*max_trans)
    newwls = jnp.array(wls[_sel])
    newtrans = jnp.array(new_trans[_sel])
    return ident, wls[_sel], new_trans[_sel]
    #np.savetxt(os.path.join(save_dir, f'{ident}.par'), np.column_stack( (wls[_sel], new_trans[_sel]) ) )
    #return ident, sedpy.observate.Filter(f'{ident}', directory=save_dir)

def to_df(filters_arr, wls=np.arange(1000., 10000., 10.), save=False, name=None):
    df_filt = pd.DataFrame(index=wls)
    for filt in filters_arr:
        f_transmit = lambda x : jnp.interp(x, filt.wavelengths, filt.transmit, left=0., right=0., period=None) 
        df_filt[filt.id] = f_transmit(wls)
    if save:
        try:
            assert name is not None, "Please specify a name for filters dataframe file."
            df_filt.to_pickle(name+'.pkl')
        except AssertionError:
            print("No name provided. Filters dataframe not written on disc.")
            pass
    return df_filt

def return_updated_filter(filt, **kwargs):
    as_dict = filt.toDict()
    as_dict.update(kwargs)
    new_filt = Filter(None, None, None, filtdict=as_dict)
    return new_filt

#################################################
# THE FOLLOWING IS ADAPTED FROM sedpy.observate #
#################################################

#try:
#    from pkg_resources import resource_filename, resource_listdir
#except(ImportError):
#    pass
#from sedpy.reference_spectra import vega, solar, sedpydir


#__all__ = ["Filter", "FilterSet", "load_filters", "list_available_filters", "getSED", "air2vac", "vac2air", "Lbol"]


def noJit_get_properties(filtwave, filttransm):
    """Determine and store a number of properties of the filter and store
    them in the object.  These properties include several 'effective'
    wavelength definitions and several width definitions, as well as the
    in-band absolute AB solar magnitude, the Vega and AB reference
    zero-point detector signal, and the conversion between AB and Vega
    magnitudes.

    See Fukugita et al. (1996) AJ 111, 1748 for discussion and definition
    of many of these quantities.
    """
    # Calculate some useful integrals
    i0 = jnp.trapz(filttransm * jnp.log(filtwave), x=jnp.log(filtwave))
    i1 = jnp.trapz(filttransm, x=jnp.log(filtwave))
    i2 = jnp.trapz(filttransm * filtwave, x=filtwave)
    i3 = jnp.trapz(filttransm, x=filtwave)

    wave_effective = jnp.exp(i0 / i1)
    wave_pivot = jnp.sqrt(i2 / i1)
    wave_mean = wave_effective
    wave_average = i2 / i3
    rectangular_width = i3 / jnp.max(filttransm)

    i4 = jnp.trapz(filttransm * jnp.power((jnp.log(filtwave / wave_effective)), 2.0), x=jnp.log(filtwave))
    gauss_width = jnp.power((i4 / i1), 0.5)
    effective_width = (2.0 * jnp.sqrt(2. * jnp.log(2.)) * gauss_width * wave_effective)

    # Get zero points and AB to Vega conversion
    ab_zero_counts = obj_counts_hires(filtwave, filttransm, filtwave, ab_gnu * lightspeed / filtwave**2)
    return ab_zero_counts

@jit
def get_properties(filtwave, filttransm):
    """Determine and store a number of properties of the filter and store
    them in the object.  These properties include several 'effective'
    wavelength definitions and several width definitions, as well as the
    in-band absolute AB solar magnitude, the Vega and AB reference
    zero-point detector signal, and the conversion between AB and Vega
    magnitudes.

    See Fukugita et al. (1996) AJ 111, 1748 for discussion and definition
    of many of these quantities.
    """
    # Calculate some useful integrals
    i0 = jnp.trapz(filttransm * jnp.log(filtwave), x=jnp.log(filtwave))
    i1 = jnp.trapz(filttransm, x=jnp.log(filtwave))
    i2 = jnp.trapz(filttransm * filtwave, x=filtwave)
    i3 = jnp.trapz(filttransm, x=filtwave)

    wave_effective = jnp.exp(i0 / i1)
    wave_pivot = jnp.sqrt(i2 / i1)
    wave_mean = wave_effective
    wave_average = i2 / i3
    rectangular_width = i3 / jnp.max(filttransm)

    i4 = jnp.trapz(filttransm * jnp.power((jnp.log(filtwave / wave_effective)), 2.0), x=jnp.log(filtwave))
    gauss_width = jnp.power((i4 / i1), 0.5)
    effective_width = (2.0 * jnp.sqrt(2. * jnp.log(2.)) * gauss_width * wave_effective)

    # Get zero points and AB to Vega conversion
    ab_zero_counts = obj_counts_hires(filtwave, filttransm, filtwave, ab_gnu * lightspeed / filtwave**2)
    return ab_zero_counts

    '''
    # If blue enough get AB mag of vega
    if wave_mean < 1e6:
        vega_zero_counts = obj_counts_hires(filtwave, filttransm, vega[:, 0], vega[:, 1])
        _ab_to_vega = -2.5 * jnp.log10(ab_zero_counts / vega_zero_counts)
    else:
        vega_zero_counts = float('NaN')
        _ab_to_vega = float('NaN')
    # If blue enough get absolute solar magnitude
    if wave_mean < 1e5:
        solar_ab_mag = ab_mag(filtwave, filt_trans, solar[:,0], solar[:,1])
    else:
        solar_ab_mag = float('NaN')
    '''
            
    '''
    @property
    def ab_to_vega(self):
        """The conversion from AB to Vega systems for this filter.  It has the
        sense

        :math:`m_{Vega} = m_{AB} + Filter().ab_to_vega`
        """
        return self._ab_to_vega
    '''
    

def noJit_obj_counts_hires(filtwave, filt_trans, sourcewave, sourceflux):
    """Project source spectrum onto filter and return the detector signal.

    Parameters
    ----------
    sourcewave : ndarray of shape ``(N_pix,)``
        Spectrum wavelength (in Angstroms). Must be monotonic increasing.

    sourceflux : ndarray of shape ``(N_source, N_pix)``
        Associated flux (assumed to be in erg/s/cm^2/AA)

    Returns
    -------
    counts : ndarray of shape ``(N_source,)``
        Detector signal(s).
    """
    # Interpolate filter transmission to source spectrum
    newtrans = jnp.interp(sourcewave, filtwave, filt_trans, left=0., right=0., period=None)

    # Integrate lambda*f_lambda*R
    counts = jnp.trapz(sourcewave * newtrans * sourceflux, x=sourcewave)
    return counts
    
@jit
def obj_counts_hires(filtwave, filt_trans, sourcewave, sourceflux):
    """Project source spectrum onto filter and return the detector signal.

    Parameters
    ----------
    sourcewave : ndarray of shape ``(N_pix,)``
        Spectrum wavelength (in Angstroms). Must be monotonic increasing.

    sourceflux : ndarray of shape ``(N_source, N_pix)``
        Associated flux (assumed to be in erg/s/cm^2/AA)

    Returns
    -------
    counts : ndarray of shape ``(N_source,)``
        Detector signal(s).
    """
    # Interpolate filter transmission to source spectrum
    newtrans = jnp.interp(sourcewave, filtwave, filt_trans, left=0., right=0., period=None)

    # Integrate lambda*f_lambda*R
    counts = jnp.trapz(sourcewave * newtrans * sourceflux, x=sourcewave)
    return counts


def noJit_ab_mag(filtwave, filt_trans, sourcewave, sourceflux):
    """Project source spectrum onto filter and return the AB magnitude.

    Parameters
    ----------
    sourcewave : ndarray of shape ``(N_pix,)``
        Spectrum wavelength (in Angstroms). Must be monotonic increasing.

    sourceflux : ndarray of shape ``(N_source, N_pix)``
        Associated flux (assumed to be in erg/s/cm^2/AA)

    Returns
    -------
    mag : float or ndarray of shape ``(N_source,)``
        AB magnitude of the source(s).
    """
    ab_zero_counts =  noJit_get_properties(filtwave, filt_trans)
    print(f'AB-counts={ab_zero_counts}')
    counts = noJit_obj_counts_hires(filtwave, filt_trans, sourcewave, sourceflux)
    print(f'filter counts={counts}')
    return -2.5 * jnp.log10(counts / ab_zero_counts)

@jit
def ab_mag(filtwave, filt_trans, sourcewave, sourceflux):
    """Project source spectrum onto filter and return the AB magnitude.

    Parameters
    ----------
    sourcewave : ndarray of shape ``(N_pix,)``
        Spectrum wavelength (in Angstroms). Must be monotonic increasing.

    sourceflux : ndarray of shape ``(N_source, N_pix)``
        Associated flux (assumed to be in erg/s/cm^2/AA)

    Returns
    -------
    mag : float or ndarray of shape ``(N_source,)``
        AB magnitude of the source(s).
    """
    ab_zero_counts =  get_properties(filtwave, filt_trans)
    counts = obj_counts_hires(filtwave, filt_trans, sourcewave, sourceflux)
    return -2.5 * jnp.log10(counts / ab_zero_counts)

'''
@jit                                   
def vega_mag(filtwave, filt_trans, sourcewave, sourceflux):
    """Project source spectrum onto filter and return the Vega magnitude.

    Parameters
    ----------
    sourcewave : ndarray of shape ``(N_pix,)``
        Spectrum wavelength (in Angstroms). Must be monotonic increasing.

    sourceflux : ndarray of shape ``(N_source, N_pix)``
        Associated flux (assumed to be in erg/s/cm^2/AA)

    Returns
    -------
    mag : float or ndarray of shape ``(N_source,)``
        Vega magnitude of the source(s).
    """
    counts = obj_counts_hires(filtwave, filt_trans, sourcewave, sourceflux)
    return -2.5 * jnp.log10(counts / self.vega_zero_counts)
'''

##########################################
# Fin de l'adaptation de sedpy.observate #
##########################################

'''
## Functions inherited from notebooks created for FORS2 analysis ##
## Temporarily saved here for use if necessary for tests or whatever ##

# mean WL (AA), full width at half maximum (AA), flux(lambda) for 0-magnitude (W/m²)
U_band_center, U_band_FWHM, U_band_f0 = 3650., 660., 3.981e-02
B_band_center, B_band_FWHM, B_band_f0 = 4450., 940., 6.310e-02
V_band_center, V_band_FWHM, V_band_f0 = 5510., 880., 3.631e-02
R_band_center, R_band_FWHM, R_band_f0 = 6580., 1380., 2.239e-02
I_band_center, I_band_FWHM, I_band_f0 = 8060., 1490., 1.148e-02

# FWHM = 2.sqrt(2.ln2).sigma for a normal distribution
U_band_sigma = U_band_FWHM/(2*np.sqrt(2*np.log(2)))
B_band_sigma = B_band_FWHM/(2*np.sqrt(2*np.log(2)))
V_band_sigma = V_band_FWHM/(2*np.sqrt(2*np.log(2)))
R_band_sigma = R_band_FWHM/(2*np.sqrt(2*np.log(2)))
I_band_sigma = I_band_FWHM/(2*np.sqrt(2*np.log(2)))

gauss_bands_dict = {\
                    "U":{"Mean": U_band_center,\
                         "Sigma": U_band_sigma,\
                         "f_0": U_band_f0\
                        },\
                    "B":{"Mean": B_band_center,\
                         "Sigma": B_band_sigma,\
                         "f_0": B_band_f0\
                        },\
                    "V":{"Mean": V_band_center,\
                         "Sigma": V_band_sigma,\
                         "f_0": V_band_f0\
                        },\
                    "R":{"Mean": R_band_center,\
                         "Sigma": R_band_sigma,\
                         "f_0": R_band_f0\
                        },\
                    "I":{"Mean": I_band_center,\
                         "Sigma": I_band_sigma,\
                         "f_0": I_band_f0\
                        }\
                   }

rect_bands_dict = {\
                   "U":{"Mean": U_band_center,\
                        "Width": U_band_FWHM,\
                        "f_0": U_band_f0\
                       },\
                   "B":{"Mean": B_band_center,\
                        "Width": B_band_FWHM,\
                        "f_0": B_band_f0\
                       },\
                   "V":{"Mean": V_band_center,\
                        "Width": V_band_FWHM,\
                        "f_0": V_band_f0\
                       },\
                   "R":{"Mean": R_band_center,\
                        "Width": R_band_FWHM,\
                        "f_0": R_band_f0\
                       },\
                   "I":{"Mean": I_band_center,\
                        "Width": I_band_FWHM,\
                        "f_0": I_band_f0\
                       }\
                  }

def gaussian_band(mu, sig, interp_step=1.):
    # Returns A FUNCTION that is created by 1D-interpolation of a normal distrib of mean mu and std dev sig
    # Interpolation range is defined arbitrarily as +/- 10sig
    # interpolation step is 1. (design case is we are working with angstrom units with a resolution of .1nm)
    _x = np.arange(mu-10*sig, mu+10*sig+interp_step, interp_step)
    _y = np.exp(-np.power(_x-mu, 2)/(2*np.power(sig, 2))) / (sig*np.power(2*np.pi, 0.5))
    _max = np.amax(_y)
    _y = _y/_max
    #_int = np.trapz(_y, _x)
    func = interp1d(_x, _y, bounds_error=False, fill_value=0.)
    return func

def rect_band(mu, width, interp_step=1.):
    # Returns A FUNCTION that is created by 1D-interpolation of a normal distrib of mean mu and std dev sig
    # Interpolation range is defined arbitrarily as +/- 10sig
    # interpolation step is 1. (design case is we are working with angstrom units with a resolution of .1nm)
    _x = np.arange(mu-width/2, mu+width/2+interp_step, interp_step)
    #_int = np.trapz(_y, _x)
    func = interp1d(_x, np.ones_like(_x), bounds_error=False, fill_value=0.)
    return func

def flux_in_band(wavelengths, spectrum, band_name, band_shape="window"):
    from astropy import constants as const
    if band_shape == "gaussian":
        _band = gaussian_band(band_name["Mean"], band_name["Sigma"])
    else:
        band_shape = "window"
        _band = rect_band(band_name["Mean"], band_name["Width"])
    _transm = spectrum * _band(wavelengths) # * ( const.c.value / np.power(wavelengths, 2) ) if flux-density is in f_nu(lambda) 
    flux = np.trapz(_transm, wavelengths)/(4*np.pi*(10.)**2) #(4*np.pi*(3.0857e17)**2) # DL=10pcs for absolute magnitudes
    return flux

def mag_in_band(wavelengths, spectrum, band_name):
    _flux = flux_in_band(wavelengths, spectrum, band_name)
    _mag0 = -2.5*np.log10(band_name["f_0"])
    mag = -2.5*np.log10(_flux) - _mag0
    return mag

def color_index(wavelengths, spectrum, band_1, band_2):
    _mag1 = mag_in_band(wavelengths, spectrum, band_1)
    _mag2 = mag_in_band(wavelengths, spectrum, band_2)
    color = _mag1 - _mag2
    return color
'''
