#!/bin/env python3

import numpy as np
from scipy.interpolate import interp1d
import os
import pandas as pd
import matplotlib.pyplot as plt

class Filter:
    """Class for filters implementation."""
    def __init__(self, name, filterfile, trans_type):
        self.name = name
        self.path = os.path.abspath(filterfile)
        self.wavelengths, self.transmit = np.loadtxt(self.path, unpack=True)
        self.trans_type = trans_type
        self.sort()
        self.transform()
        self.clip_filter()
        self.f_transmit = interp1d(self.wavelengths, self.transmit, bounds_error=False, fill_value=0.)
        #f,a = plt.subplots(1,1)
        #a.plot(self.wavelengths, self.f_transmit(self.wavelengths))
        #f.show()
        
    def __str__(self):
        return f"{self.name} filter, from file : {self.path}."
        
    def __repr__(self):
        return f"<Filter object : name={self.name} ; path={self.path}>"
        
    def sort(self):
        _inds = np.argsort(self.wavelengths)
        self.wavelengths = self.wavelengths[_inds]
        self.transmit = self.transmit[_inds]
        del _inds
        
    def lambda_mean(self):
        self.mean_wl = np.trapz(self.wavelengths*self.transmit, self.wavelengths) / np.trapz(self.transmit, self.wavelengths)
        
    def clip_filter(self):
        _indToKeep = np.where(self.transmit >= 0.01*np.max(self.transmit))[0]
        self.wavelengths = self.wavelengths[ _indToKeep ]
        self.transmit = self.transmit[ _indToKeep ]
        _lambdas_peak = self.wavelengths[ np.where(self.transmit >= 0.5*np.max(self.transmit))[0] ]
        self.min_wl, self.max_wl = np.min(self.wavelengths), np.max(self.wavelengths)
        self.minWL_peak, self.maxWL_peak = np.min(_lambdas_peak), np.max(_lambdas_peak)
        self.peak_wl = self.wavelengths[ np.argmax(self.transmit) ]
        
    def transform(self):
        self.lambda_mean()
        if (self.trans_type.lower() == "photons") and (self.mean_wl>0.):
            for k, _wl in enumerate(self.wavelengths):
                self.transmit[k] = self.transmit[k]*(_wl/self.mean_wl)


def to_df(filters_arr, wls=np.arange(1000., 10000., 10.), save=False, name=None):
    df_filt = pd.DataFrame(index=wls)
    for filt in filters_arr:
        df_filt[filt.name] = filt.f_transmit(wls)
    if save:
        try:
            assert name is not None, "Please specify a name for filters dataframe file."
            df_filt.to_pickle(name+'.pkl')
        except AssertionError:
            print("No name provided. Filters dataframe not written on disc.")
            pass
    return df_filt







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
