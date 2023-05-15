#!/bin/env python3

import numpy as np
from EmuLP import Filter
from copy import deepcopy
import pandas as pd

class Galaxy:
    """
    Galaxy objects corresponding to each observation in the dataset.\n
    Photometry to be provided as a single array of length 2*n with n the number of filter in argument `photometry`\n
    and structured as MEME or FEFE (mag, error, etc. or flux, error, etc.).\n
    The type of photometry is to be specified as F or M in the argument `struct`.
    """
    
    def __init__(self, ident, photometry, ismag, zs=None):
        assert len(photometry)%2 == 0, "Missing data in observations : check that magnitudes/fluxes and errors are available\n and listed as M (or F), error, M (or F), error, etc."
        _phot = np.array([])
        _phot_errs = np.array([])
        
        self.id = f"{ident}" # name or number, but as a string
        if zs is None:
            self.zs = None
            self.is_spectro = False
        else :
            self.zs = zs
            self.is_spectro = True
        
        for i in range(len(photometry)//2):
            _phot = np.append(_phot, photometry[2*i])
            _phot_errs = np.append(_phot_errs, photometry[2*i+1])
        
        if not ismag:
            #self.obs_type = "fluxes"
            self.magnitudes = -2.5*np.log10(_phot) - 48.6 # conversion to AB-magnitudes.
            self.mag_errors = -2.5*np.log10(1+_phot_errs/_phot) # to be checked.
        else :
            #self.obs_type = "AB-magnitudes"
            self.magnitudes = _phot
            self.mag_errors = _phot_errs

    def __str__(self):
        if self.is_spectro:
            _str = f"Galaxy {self.id} contains {len(self.magnitudes)} observations, true redshift is {self.zs}."
        else:
            _str = f"Galaxy {self.id} contains {len(self.magnitudes)} observations, true redshift is unknown."
        return _str
        
    def __repr__(self):
        if self.is_spectro:
            _str = f"<Galaxy object : id={self.id} ; z_spec={self.zs}>"
        else:
            _str = f"<Galaxy object : id={self.id} ; unknown z_spec>"
        return _str
        
    def estimate_zp(self, estimator, filters):
        self.zp, self.template, self.extinction_law, self.ebv, self.chi2 = estimator.estimate_zp(self, filters)
        
    def chi2(self, filters, template, scale=None, rescale_all=False):
        if (scale is None) or rescale_all:
            _scaled_temp, _scale = template.auto_scale(self, filters, fast=True)
        else:
            _scaled_temp = deepcopy(template)
            _scaled_temp.rescale(scale)
            _scale = deepcopy(scale)
            
        _sel1 = np.isfinite(self.magnitudes)
        _sel2 = np.isfinite(template.magnitudes)
        _sel3 = np.isfinite(self.mag_errors)
        _sel = [ (b1 and b2 and b3) for b1,b2,b3 in zip(_sel1, _sel2, _sel3) ]
        
        if len(_scaled_temp.magnitudes[_sel]) > 0 :
            _terms = np.array([ np.power((_mag - _scaled_temp_mag)/_err, 2.)\
                                for _mag,_err,_scaled_temp_mag in zip(self.magnitudes[_sel], self.mag_errors[_sel], _scaled_temp.magnitudes[_sel])\
                                ])
            _chi2 = np.sum(_terms)/len(_scaled_temp.magnitudes[_sel])
        else :
            _chi2 = np.inf
        del _scaled_temp
        return _chi2, _scale
        
def to_df(galaxies_arr, filters_arr, save=False, name=None):
    df_gal = pd.DataFrame(index=[_gal.id for _gal in galaxies_arr])
    for i,filt in enumerate(filters_arr):
        df_gal[f"Mag({filt.name})"] = [ _gal.magnitudes[i] for _gal in galaxies_arr ]
        df_gal[f"MagErr({filt.name})"] = [ _gal.mag_errors[i] for _gal in galaxies_arr ]
    df_gal["Photometric redshift"] = [ _gal.zp for _gal in galaxies_arr ]
    df_gal["True redshift"] = [_gal.zs if _gal.is_spectro else None for _gal in galaxies_arr]
    df_gal["Template SED"] = [ _gal.template for _gal in galaxies_arr ]
    df_gal["Extinction law"] = [_gal.extinction_law for _gal in galaxies_arr]
    df_gal["E(B-V)"] = [_gal.ebv for _gal in galaxies_arr]
    df_gal["Chi2"] = [_gal.chi2 for _gal in galaxies_arr]
    
    if save:
        try:
            assert name is not None, "Please specify a name for galaxies dataframe file."
            df_gal.to_pickle(name+'.pkl')
        except AssertionError:
            print("No name provided. Galaxies dataframe not written on disc.")
            pass
    return df_gal
