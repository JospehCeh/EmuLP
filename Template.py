#!/bin/env python3

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar, minimize
import os, sys
from EmuLP import Cosmology
from EmuLP import Extinction
from EmuLP import Filter
from copy import deepcopy

class Template:
    """SED Templates to be used for photo-z estimation"""
    
    def __init__(self, name, specfile):
        self.name = name
        self.path = os.path.abspath(specfile)
        self.scaling_factor = 1.
        self.wavelengths, self.lumins = np.loadtxt(self.path, unpack=True)
        self.f_lum = interp1d(self.wavelengths, self.lumins, bounds_error=False, fill_value=0.)
        self.ebv = 0.
        self.extinc_law = None
        self.redshift = 0.
    
    def __str__(self):
        return f"SED template {self.name} from {self.path} at z={self.redshift} ; extinguished IAW : {self.extinc_law} with E-B(V)={self.ebv}."
        
    def __repr__(self):
        return f"<Template object : name={self.name} ; spectrum={self.path} ; z={self.redshift} ; extinction={self.extinc_law} ; E-B(V)={self.ebv}>"
    
    def rescale(self, factor):
        self.scaling_factor = factor
        self.lumins = self.scaling_factor*self.lumins
        self.f_lum = interp1d(self.wavelengths, self.lumins, bounds_error=False, fill_value=0.)
        self.magnitudes = self.magnitudes - 2.5*np.log10(self.scaling_factor)
        
    def unscale(self):
        self.lumins = self.lumins / self.scaling_factor
        self.magnitudes = self.magnitudes + 2.5*np.log10(self.scaling_factor)
        self.f_lum = interp1d(self.wavelengths, self.lumins, bounds_error=False, fill_value=0.)
        self.scaling_factor = 1.
    
    def apply_cosmo(self, cosmo, z):
        self.cosmo = cosmo
        self.d_modulus = cosmo.distMod(self.redshift) # Mpc
        self.d_metric = cosmo.distMet(self.redshift) # Mpc
        self.d_luminosity = cosmo.distLum(self.redshift) # Mpc
        self.d_angular = cosmo.distAng(self.redshift) # Mpc
        self.age_univ = cosmo.time(self.redshift) # years
        
    def auto_apply_cosmo(self):
        if not ((self.cosmo is None) or (self.redshift is None)):
            self.apply_cosmo(self.cosmo, self.redshift)
        else:
            raise RuntimeError("Template need a cosmology and/or a redshift to be applied. Check object creation and previous manipulations.")

    def normalize(self, wl_inf, wl_sup):
        _sel = (self.wavelengths>=wl_inf)*(self.wavelengths<=wl_sup)
        _wls = self.wavelengths[_sel]
        self.norm = np.trapz(self.f_lum(_wls), _wls)
        self.spec_norm = self.lumins / self.norm
    
    def to_redshift(self, z, cosmo=None):
        self.wavelengths = self.wavelengths * (1.+z)
        self.redshift = z
        self.f_lum = interp1d(self.wavelengths, self.lumins, bounds_error=False, fill_value=0.)
        if not (cosmo is None):
            self.apply_cosmo(cosmo, z)
    
    def to_restframe(self):
        self.wavelengths = self.wavelengths / (1.+self.redshift)
        self.redshift = 0.
        if not (self.cosmo is None):
            self.apply_cosmo(self.cosmo, 0.)
        
    def apply_extinc(self, law, ebv):
        self.extinc_law = law
        self.ebv = ebv
        ext_func = self.extinc_law.extinct_func(self.wavelengths, self.ebv)
        new_func = interp1d(self.wavelengths, self.f_lum(self.wavelengths)*ext_func(self.wavelengths), bounds_error=False, fill_value=0.)
        self.f_lum = new_func
        
    def compute_flux(self, filt):
        flux = np.trapz(self.f_lum(self.wavelengths)*1.0e23 * filt.f_transmit(self.wavelengths), self.wavelengths)/(4*np.pi*np.power(self.d_luminosity*3.0857e22, 2)) # Unit of d_luminosity? parsecs or Mpc?
        return flux
    
    def compute_magAB(self, filt):
        flux = self.compute_flux(filt)
        mag = -2.5*np.log10(flux) - 48.6
        return mag
        
    def fill_magnitudes(self, filters):
        self.magnitudes = np.array([ self.compute_magAB(_f) for _f in filters ])
    
    def auto_scale(self, gal, filts, fast=True):
        scaled_template = deepcopy(self)
        
        _sel1 = np.isfinite(gal.magnitudes)
        _sel2 = np.isfinite(self.magnitudes)
        _sel3 = np.isfinite(gal.mag_errors)
        _sel = [ (b1 and b2 and b3) for b1,b2,b3 in zip(_sel1, _sel2, _sel3) ]
        
        if len(scaled_template.magnitudes[_sel]) > 0 :
            if fast:
                #print(gal.magnitudes[_sel], self.magnitudes[_sel])
                _scale = np.mean(np.abs(gal.magnitudes[_sel]-self.magnitudes[_sel]))
                #print(_scale)
                scaled_template.rescale(_scale)
            else:
                def chi2(a):
                    scaled_template.rescale(a)
                    #_mags = np.array([scaled_template.compute_magAB(_filt) for _filt in filts])
                    _terms = np.power((gal.magnitudes[_sel] - scaled_template.magnitudes[_sel])/gal.mag_errors[_sel], 2.)
                    _chi2 = np.sum(_terms)
                    return _chi2
                #optim = minimize_scalar(chi2)
                optim = minimize(chi2, x0=np.abs((gal.magnitudes[0]-self.magnitudes[0])))
                _scale = optim.x
                scaled_template.rescale(_scale)
        else:
            _scale = 1.
        return scaled_template, _scale

def to_df(templates_arr, filters_arr, wls=np.arange(1000., 10000., 10.), save=False, name=None):
    df_spec = pd.DataFrame(index=wls)
    df_prop = pd.DataFrame(index=[_templ.name for _templ in templates_arr])
    for templ in templates_arr:
        df_spec[templ.name] = templ.f_lum(wls)
        templ.normalize(np.min(wls), np.max(wls))
        df_spec[templ.name+"_normed"] = interp1d(templ.wavelengths, templ.spec_norm, bounds_error=False, fill_value=0.)(wls)
    df_prop["Path"] = [_templ.path for _templ in templates_arr]
    df_prop["Redshift"] = [_templ.redshift for _templ in templates_arr]
    df_prop["Extinction law"] = [_templ.extinc_law for _templ in templates_arr]
    df_prop["E(B-V)"] = [_templ.ebv for _templ in templates_arr]
    df_prop["Cosmology"] = [_templ.cosmo for _templ in templates_arr]
    df_prop["Luminosity distance"] = [_templ.d_luminosity for _templ in templates_arr]
    df_prop["Lookback time"] = [_templ.age_univ for _templ in templates_arr]
    for i,filt in enumerate(filters_arr):
        df_prop[f"Mag({filt.name})"] = [ _templ.magnitudes[i] for _templ in templates_arr ]
    
    if save:
        try:
            assert name is not None, "Please specify a name for templates dataframe files."
            df_spec.to_pickle(name+'.pkl')
            df_prop.to_pickle(name+"_properties"+'.pkl')
        except AssertionError:
            print("No name provided. Templates dataframes not written on disc.")
            pass
    return df_spec, df_prop
