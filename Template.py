#!/bin/env python3

from functools import partial
from jax import vmap, pmap, jit, debug
import jax.numpy as jnp
import numpy as np
import pandas as pd
#from scipy.interpolate import interp1d
#from scipy.optimize import minimize_scalar, minimize, Bounds
import os, sys
from EmuLP import Cosmology
from EmuLP import Extinction
from EmuLP import Filter
from copy import deepcopy
import munch

class Template(munch.Munch):
    """SED Templates to be used for photo-z estimation"""
    
    def __init__(self, ident, specfile, specdict=None):
        if specdict is None:
            super().__init__()
            self.id = ident
            #self.path = os.path.abspath(specfile)
            self.scaling_factor = 1.
            self.wavelengths, self.lumins = np.loadtxt(os.path.abspath(specfile), unpack=True)
            self.lumins = np.array([l if l>0. else 1.0e-20 for l in self.lumins])
            #self.f_lum = lambda x : jnp.interp(x, self.wavelengths, self.lumins, left=0., right=0., period=None)
            #interp1d(self.wavelengths, self.lumins, bounds_error=False, fill_value=0.)
            self.ebv = 0.
            self.extinc_law = 0
            self.redshift = 0.
        else:
            super().__init__(specdict)
    
    def __str__(self):
        return f"SED template {self.id} at z={self.redshift} ; extinguished IAW : {self.extinc_law} with E-B(V)={self.ebv}."
        
    def __repr__(self):
        return f"<Template object : name={self.id} ; z={self.redshift} ; extinction={self.extinc_law} ; E-B(V)={self.ebv}>"
    
    def __eq__(self, other):
        return (self.id == other.id) and (self.extinc_law == other.extinc_law) and (self.ebv == other.ebv) and (self.redshift == other.redshift)
    
    def __ne__(self, other):
        return not self.__eq__(other)
    
    def __hash__(self):
        hasval = hash((14, self.id, self.extinc_law, self.ebv, self.redshift))
        return hashval
        
    def auto_apply_cosmo(self):
        if not ((self.cosmo is None) or (self.redshift is None)):
            newself = apply_cosmo(self, self.cosmo, self.redshift)
            return newself
        else:
            raise RuntimeError("Template need a cosmology and/or a redshift to be applied. Check object creation and previous manipulations.")
            
            
    '''
    @partial(vmap, in_axes=(None, 0, None), out_axes={'age_univ': *,\
                                                      'cosmo': {'h0': *,\
                                                                'jcosmo': CustomNode(Cosmology[{'gamma_growth': False}],\
                                                                                     [*, *, *, *, *, *, *, *]),\
                                                                'l0': *,\
                                                                'om0': *,\
                                                                'omt': *},\
                                                      'd_angular': *,\
                                                      'd_luminosity': *,\
                                                      'd_metric': *,\
                                                      'd_modulus': *,\
                                                      'ebv': *,\
                                                      'extinc_law': {'coeff': *,\
                                                                     'id': *,\
                                                                     'wavelengths': *},\
                                                      'id': *,\
                                                      'lumins': *,\
                                                      'redshift': *,\
                                                      'scaling_factor': *,\
                                                      'wavelengths': *\
                                                     }\
            )
    def to_redshift(self, z, cosmo=None):
        _templ = deepcopy(self)
        _templ.wavelengths = (1.+z) * _templ.wavelengths
        _templ.redshift = z
        #_templ.f_lum = interp1d(_templ.wavelengths, _templ.lumins, bounds_error=False, fill_value=0.)
        #_templ.f_lum = lambda x : jnp.interp(x, _templ.wavelengths, _templ.lumins, left=0., right=0., period=None)
        if not (cosmo is None):
            _templ.apply_cosmo(cosmo, z)
        return munch.unmunchify(_templ)
    '''
    
    def compute_magAB(self, filt):
        f_lum = lambda x : jnp.interp(x, self.wavelengths, self.lumins, left=0., right=0., period=None)
        mag = filt.sedpy_filt.ab_mag(self.wavelengths, f_lum(self.wavelengths)) + self.d_modulus
        return mag
    
    def auto_scale(self, gal, filts):
        scaled_template = deepcopy(self)
        
        _sel1 = np.isfinite(gal.f_ab)
        _sel2 = np.isfinite(self.f_ab)
        _sel3 = np.isfinite(gal.f_ab_err)
        _sel = [ (b1 and b2 and b3) for b1,b2,b3 in zip(_sel1, _sel2, _sel3) ]
        
        scaled_template.fill_magnitudes(filts)
        if len(scaled_template.f_ab[_sel]) > 0 :
            # Scaling as in LEPHARE
            avmago = np.sum(gal.f_ab[_sel]*self.f_ab[_sel]/np.power(gal.f_ab_err[_sel], 2.))
            avmagt = np.sum(np.power(self.f_ab[_sel]/gal.f_ab_err[_sel], 2.))
            _scale = avmago/avmagt
            scaled_template.rescale(_scale)
            scaled_template.fill_magnitudes(filts)
        else:
            _scale = 1.
            scaled_template.fill_magnitudes(filts)
        return scaled_template, _scale


def make_base_template(ident, specfile, wl_grid):
    wl, _lums = np.loadtxt(os.path.abspath(specfile), unpack=True)
    _inds = jnp.argsort(wl)
    wls = wl[_inds]
    lum = _lums[_inds]
    wavelengths, lums = jnp.array(wls), jnp.array([l if l>0. else 1.0e-20 for l in lum])
    lumins = jnp.interp(wl_grid, wavelengths, lums, left=0., right=0., period=None)
    return ident, lumins

def nojit_no_ext_make_template(base_temp_lums, filts, z, cosmo, wl_grid):
    lumins = base_temp_lums
    zshift_wls = wl_grid*(1.+z) #jnp.interp(wl_grid, wavelengths, wavelengths*(1.+z), left=0., right=0., period=None)
    d_modulus = Cosmology.calc_distMod(cosmo, z)
    print(f"Dist. modulus = {d_modulus}")
    mags = jnp.array([Filter.noJit_ab_mag(filt.wavelengths, filt.transmission, zshift_wls, lumins) + d_modulus\
                      for filt in filts])
    #f_ab = jnp.power(10., -0.4*(mags+48.6))
    return mags

def nojit_make_template(base_temp_lums, filts, extinc_arr, z, cosmo, wl_grid):
    lumins = base_temp_lums
    ext_lumins = lumins*extinc_arr
    zshift_wls = wl_grid*(1.+z) #jnp.interp(wl_grid, wavelengths, wavelengths*(1.+z), left=0., right=0., period=None)
    d_modulus = Cosmology.calc_distMod(cosmo, z)
    mags = jnp.array([Filter.ab_mag(filt.wavelengths, filt.transmission, zshift_wls, ext_lumins) + d_modulus\
                      for filt in filts])
    #f_ab = jnp.power(10., -0.4*(mags+48.6))
    return mags

def nojit_make_scaled_template(base_temp_lums, filts, extinc_arr, z, cosmo, galax_fab, galax_fab_err, wl_grid):
    lumins = base_temp_lums
    ext_lumins = lumins*extinc_arr
    zshift_wls = wl_grid*(1.+z) #jnp.interp(wl_grid,  wavelengths, wavelengths*(1.+z), left=0., right=0., period=None)
    d_modulus = Cosmology.calc_distMod(cosmo, z)
    mags = jnp.array([Filter.ab_mag(filt.wavelengths, filt.transmission, zshift_wls, ext_lumins) + d_modulus\
                      for filt in filts])
    f_ab = jnp.power(10., -0.4*(mags+48.6))
    
    scale = calc_scale_arrs(f_ab, galax_fab, galax_fab_err)
    print(f"Scale={scale}")
    scaled_lumins = ext_lumins*scale
    scaled_mags = jnp.array([Filter.ab_mag(filt.wavelengths, filt.transmission, zshift_wls, scaled_lumins) + d_modulus\
                             for filt in filts])
    scaled_f_ab = jnp.power(10., -0.4*(scaled_mags+48.6))
    return scaled_mags

@partial(jit, static_argnums=(0,4))
#@partial(vmap, in_axes=(None, None, 0, 0, None, None))
def make_template(base_temp_lums, filts, extinc_arr, z, cosmo, wl_grid):
    lumins = base_temp_lums
    ext_lumins = lumins*extinc_arr
    zshift_wls = wl_grid*(1.+z) #jnp.interp(wl_grid, wavelengths, wavelengths*(1.+z), left=0., right=0., period=None)
    #d_modulus = Cosmology.calc_distMod(cosmo, z)
    d_modulus = Cosmology.distMod(cosmo, z)
    mags = jnp.array([Filter.ab_mag(filt.wavelengths, filt.transmission, zshift_wls, ext_lumins) + d_modulus\
                      for filt in filts])
    f_ab = jnp.power(10., -0.4*(mags+48.6))
    return f_ab

@partial(jit, static_argnums=4)
#@partial(vmap, in_axes=(None, None, 0, 0, None, 0, 0, None))
def make_scaled_template(base_temp_lums, filts, extinc_arr, z, cosmo, galax_fab, galax_fab_err, wl_grid):
    lumins = base_temp_lums
    ext_lumins = lumins*extinc_arr
    zshift_wls = wl_grid*(1.+z) #jnp.interp(wl_grid,  wavelengths, wavelengths*(1.+z), left=0., right=0., period=None)
    #d_modulus = Cosmology.calc_distMod(cosmo, z)
    d_modulus = Cosmology.distMod(cosmo, z)
    mags = jnp.array([Filter.ab_mag(filt.wavelengths, filt.transmission, zshift_wls, ext_lumins) + d_modulus\
                      for filt in filts])
    f_ab = jnp.power(10., -0.4*(mags+48.6))
    
    scale = calc_scale_arrs(f_ab, galax_fab, galax_fab_err)
    scaled_lumins = ext_lumins*scale
    scaled_mags = jnp.array([Filter.ab_mag(filt.wavelengths, filt.transmission, zshift_wls, scaled_lumins) + d_modulus\
                             for filt in filts])
    scaled_f_ab = jnp.power(10., -0.4*(scaled_mags+48.6))
    return scaled_f_ab

'''
@partial(jit, static_argnums=(0,4))
#@partial(vmap, in_axes=(None, None, 0, 0, None, 0, 0, None))
def make_scaled_template(specfile, filts, extinc_arr, z, cosmo, galax_fab, galax_fab_err, wl_grid):
    wls, _lums = np.loadtxt(os.path.abspath(specfile), unpack=True)
    wavelengths, lums = jnp.array(wls), jnp.array([l if l>0. else 1.0e-20 for l in _lums])
    lumins = jnp.interp(wl_grid, wavelengths, lums, left=0., right=0., period=None)
    ext_lumins = lumins*extinc_arr
    zshift_wls = wl_grid*(1.+z) #jnp.interp(wl_grid, wavelengths, wavelengths*(1.+z), left=0., right=0., period=None)
    d_modulus = Cosmology.calc_distMod(cosmo, z)
    mags = jnp.array([Filter.ab_mag(filt.wavelengths, filt.transmission, zshift_wls, ext_lumins) + d_modulus\
                      for filt in filts])
    f_ab = jnp.power(10., -0.4*(mags+48.6))
    
    scale = calc_scale_arrs(f_ab, galax_fab, galax_fab_err)
    scaled_lumins = ext_lumins*scale
    scaled_mags = jnp.array([Filter.ab_mag(filt.wavelengths, filt.transmission, zshift_wls, scaled_lumins) + d_modulus\
                             for filt in filts])
    scaled_f_ab = jnp.power(10., -0.4*(scaled_mags+48.6))
    return scaled_f_ab
'''

    
def fAB_arrays(templ):
    return templ.f_ab, jnp.array([templ.id, templ.redshift, templ.ebv, templ.extinc_law])

def mAB_arrays(templ):
    return templ.magnitudes, jnp.array([templ.id, templ.redshift, templ.ebv, templ.extinc_law])

def lumins_arrays(templ, wl_grid):
    f_lum = lambda x : jnp.interp(x, templ.wavelengths, templ.lumins, left=0., right=0., period=None)
    lumins = f_lum(wl_grid)
    return lumins

#@partial(jit, static_argnums=(0,1,2))
@jit
def calc_scale_arrs(f_templ, f_gal, err_gal):
    #_sel1 = jnp.isfinite(f_gal)
    #_sel2 = jnp.isfinite(f_templ)
    #_sel3 = jnp.isfinite(err_gal)
    #_sel = [ (b1 and b2 and b3) for b1,b2,b3 in zip(_sel1, _sel2, _sel3) ]
    #_sel = _sel1 * _sel2 * _sel3
    #if len(f_templ[_sel]) > 0 :
        # Scaling as in LEPHARE
    
    arr_o = f_gal/err_gal
    arr_t = f_templ/err_gal
    avmago = jnp.sum(arr_o*arr_t)
    avmagt = jnp.sum(jnp.power(arr_t, 2.))
    _scale = avmago/avmagt
    #else:
    #    _scale = 1.
    return _scale
    
def apply_extinc(templ, law, file, ebv):
    extinc = Extinction.Extinction(law, file)
    ext_func = extinc.extinct_func(templ.wavelengths, ebv)
    f_lum = lambda x : jnp.interp(x, templ.wavelengths, templ.lumins, left=0., right=0., period=None)
    _new_lums = f_lum(templ.wavelengths)*ext_func(templ.wavelengths)
    new_func = lambda x : jnp.interp(x, templ.wavelengths, _new_lums, left=0., right=0., period=None)
    #interp1d(self.wavelengths, self.f_lum(self.wavelengths)*ext_func(self.wavelengths), bounds_error=False, fill_value=0.)
    #self.f_lum = new_func
    return return_updated_template(templ, extinc_law=law, ebv=ebv, lumins=new_func(templ.wavelengths))

#@partial(vmap, in_axes=(None, None, 0))
def apply_cosmo(templ, cosmol, z):
    newtempl = return_updated_template(templ, cosmo=cosmol,\
                                       d_modulus = cosmol.distMod(z),\
                                       d_metric = cosmol.j_distM(z),\
                                       d_luminosity = cosmol.j_distLum(z),\
                                       d_angular = cosmol.j_distAng(z),\
                                       redshift = z) # distances in Mpc
    #self.age_univ = cosmo.time(z) # years
    return newtempl

def rescale(templ, factor):
    return return_updated_template(templ, scaling_factor=factor, lumins=factor*templ.lumins)
    #self.f_lum = lambda x : jnp.interp(x, self.wavelengths, self.lumins, left=0., right=0., period=None)
    #interp1d(self.wavelengths, self.lumins, bounds_error=False, fill_value=0.)
    #self.magnitudes = self.magnitudes - 2.5*np.log10(self.scaling_factor)

def unscale(templ):
    return return_updated_template(templ, lumins=templ.lumins/templ.scaling_factor, scaling_factor=1.)
    #self.magnitudes = self.magnitudes + 2.5*np.log10(self.scaling_factor)
    #self.f_lum = lambda x : jnp.interp(x, self.wavelengths, self.lumins, left=0., right=0., period=None)
    #interp1d(self.wavelengths, self.lumins, bounds_error=False, fill_value=0.)

def normalize(templ, wl_inf, wl_sup):
    _sel = (templ.wavelengths>=wl_inf)*(templ.wavelengths<=wl_sup)
    _wls = templ.wavelengths[_sel]
    f_lum = lambda x : jnp.interp(x, templ.wavelengths, templ.lumins, left=0., right=0., period=None)
    _norm = np.trapz(f_lum(_wls), _wls)
    return return_updated_template(templ, norm=_norm, spec_norm=templ.lumins/_norm)
    
def to_restframe(templ):
    _newtemp = return_updated_template(templ, wavelengths=templ.wavelengths/(1.+templ.redshift), redshift=0.)
    newtemp = _newtemp.auto_apply_cosmo()
    return newtemp

# Pourrait utiliser les fonctions pré-établies dans sedpy...
def fill_magnitudes(templ, filters):
    mags = jnp.array([ templ.compute_magAB(_f) for _f in filters ])
    return return_updated_template(templ, magnitudes=mags, f_ab=jnp.power(10., -0.4*(mags+48.6)))
    #self.magnitudes = -2.5*np.log10(self.f_ab) - 48.6

def to_df(templates_arr, filters_arr, wls=np.arange(1000., 10000., 10.), save=False, name=None):
    df_spec = pd.DataFrame(index=wls)
    df_prop = pd.DataFrame()
    for templ in templates_arr:
        f_lum = lambda x : jnp.interp(x, templ.wavelengths, templ.lumins, left=0., right=0., period=None)
        df_spec[f"{templ.id}"] = f_lum(wls)
        templ = normalize(templ, np.min(wls), np.max(wls))
        df_spec[f"{templ.id}_normed"] = (lambda x : jnp.interp(x, templ.wavelengths, templ.spec_norm,\
                                                              left=0., right=0., period=None))(wls)
        #interp1d(templ.wavelengths, templ.spec_norm, bounds_error=False, fill_value=0.)(wls)
    df_prop["Model"] = [_templ.id for _templ in templates_arr]
    #df_prop["Path"] = [_templ.path for _templ in templates_arr]
    df_prop["Redshift"] = [_templ.redshift for _templ in templates_arr]
    df_prop["Extinction law"] = [_templ.extinc_law for _templ in templates_arr]
    df_prop["E(B-V)"] = [_templ.ebv for _templ in templates_arr]
    df_prop["Cosmology"] = [str(_templ.cosmo) for _templ in templates_arr]
    df_prop["Luminosity distance"] = [_templ.d_luminosity for _templ in templates_arr]
    #df_prop["Lookback time"] = [_templ.age_univ for _templ in templates_arr]
    for i,filt in enumerate(filters_arr):
        df_prop[f"Mag({filt.id})"] = [ _templ.magnitudes[i] for _templ in templates_arr ]
    
    if save:
        try:
            assert name is not None, "Please specify a name for templates dataframe files."
            df_spec.to_pickle(name+'.pkl')
            df_prop.to_pickle(name+"_properties"+'.pkl')
        except AssertionError:
            print("No name provided. Templates dataframes not written on disc.")
            pass
    return df_spec, df_prop

@partial(vmap, in_axes=(None, 0, None), out_axes={'cosmo': None,\
                                                  'd_angular': 0,\
                                                  'd_luminosity': 0,\
                                                  'd_metric': 0,\
                                                  'd_modulus': 0,\
                                                  'ebv': None,\
                                                  'extinc_law': None,\
                                                  'id': None,\
                                                  'lumins': 0,\
                                                  'redshift': 0,\
                                                  'scaling_factor': None,\
                                                  'wavelengths': 0\
                                                 })
def to_redshift(temp_dict, z, cosmo=None):
    _templ = Template(None, None, specdict=temp_dict)
    _templ = return_updated_template(_templ, wavelengths=(1.+z)*_templ.wavelengths, redshift=z)
    #_templ.f_lum = interp1d(_templ.wavelengths, _templ.lumins, bounds_error=False, fill_value=0.)
    #_templ.f_lum = lambda x : jnp.interp(x, _templ.wavelengths, _templ.lumins, left=0., right=0., period=None)
    if not (cosmo is None):
        _cosm = Cosmology.Cosmology(cosmodict=cosmo)
        _templ = apply_cosmo(_templ, _cosm, z)
    return munch.unmunchify(_templ)

def return_updated_template(template, **kwargs):
    as_dict = template.toDict()
    as_dict.update(kwargs)
    new_temp = Template(None, None, specdict=as_dict)
    return new_temp

'''
#@partial(vmap, in_axes=(None, None, None, None, None))
def j_interp1d(xp, fp, l=0., r=0., p=None):
    f = lambda x : jnp.interp(x, xp, fp, left=l, right=r, period=p)
    return f
'''