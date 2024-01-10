#!/bin/env python3

from functools import partial
from jax import vmap, jit, debug
import numpy as np
import jax.numpy as jnp
#from scipy.interpolate import interp1d
#from scipy.optimize import minimize_scalar, minimize, Bounds
import os, sys
from EmuLP import Cosmology
from EmuLP import Extinction
from EmuLP import Filter
from collections import namedtuple

BaseTemplate = namedtuple('BaseTemplate', ['name', 'flux'])

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
    d_modulus = Cosmology.distMod(cosmo, z)
    print(f"Dist. modulus = {d_modulus}")
    mags = jnp.array([Filter.noJit_ab_mag(filt.wavelengths, filt.transmission, zshift_wls, lumins) + d_modulus\
                      for filt in filts])
    #f_ab = jnp.power(10., -0.4*(mags+48.6))
    return mags

def nojit_make_template(base_temp_lums, filts, extinc_arr, z, cosmo, wl_grid, opacities):
    lumins = base_temp_lums
    ext_lumins = lumins*extinc_arr*opacities
    zshift_wls = wl_grid*(1.+z) #jnp.interp(wl_grid, wavelengths, wavelengths*(1.+z), left=0., right=0., period=None)
    d_modulus = Cosmology.distMod(cosmo, z)
    mags = jnp.array([Filter.ab_mag(filt.wavelengths, filt.transmission, zshift_wls, ext_lumins) + d_modulus\
                      for filt in filts])
    #f_ab = jnp.power(10., -0.4*(mags+48.6))
    return mags

def nojit_make_scaled_template(base_temp_lums, filts, extinc_arr, z, cosmo, galax_fab, galax_fab_err, wl_grid, opacities):
    lumins = base_temp_lums
    ext_lumins = lumins*extinc_arr*opacities
    zshift_wls = wl_grid*(1.+z) #jnp.interp(wl_grid,  wavelengths, wavelengths*(1.+z), left=0., right=0., period=None)
    d_modulus = Cosmology.distMod(cosmo, z)
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

@partial(jit, static_argnums=4)
#@partial(vmap, in_axes=(None, None, 0, 0, None, None))
def make_template(base_temp_lums, filts, extinc_arr, z, cosmo, wl_grid, opacities):
    #ext_lumins = base_temp_lums*extinc_arr
    #zshift_wls = wl_grid*(1.+z) #jnp.interp(wl_grid, wavelengths, wavelengths*(1.+z), left=0., right=0., period=None)
    #d_modulus = Cosmology.calc_distMod(cosmo, z)
    #d_modulus = Cosmology.distMod(cosmo, z)
    mags = jnp.array([Filter.ab_mag(filt.wavelengths, filt.transmission, wl_grid*(1.+z),\
                                    base_temp_lums*extinc_arr*opacities) + Cosmology.distMod(cosmo, z)\
                      for filt in filts])
    return jnp.power(10., -0.4*(mags+48.6))

@jit
def make_dusty_template(base_temp_lums, filts, extinc_arr, wl_grid):
    #ext_lumins = calc_dusty_transm(base_temp_lums, extinc_arr)
    mags = jnp.array([Filter.ab_mag(filt.wavelengths, filt.transmission, wl_grid, calc_dusty_transm(base_temp_lums, extinc_arr))\
                      for filt in filts])
    return jnp.power(10., -0.4*(mags+48.6))

@jit
def calc_dusty_transm(base_temp_lums, extinc_arr):
    return base_temp_lums*extinc_arr

@jit
def calc_fab(filts, wvls, lums, d_mod=0.):
    mags = jnp.array([Filter.ab_mag(filt.wavelengths, filt.transmission, wvls, lums) + d_mod\
                      for filt in filts])
    return jnp.power(10., -0.4*(mags+48.6))

@jit
def calc_nuvk(baseTemp_flux, extLaw_transm, wlgrid):
    #dusty_trans = calc_dusty_transm(baseTemp_flux, extLaw_transm)
    mab_NUV, mab_NIR = -2.5*jnp.log10(calc_fab((Filter.NUV_filt, Filter.NIR_filt), wlgrid, calc_dusty_transm(baseTemp_flux, extLaw_transm), d_mod=0.))-48.6
    return mab_NUV-mab_NIR

@jit
def make_scaled_template(base_temp_lums, filts, extinc_arr, galax_fab, galax_fab_err, z, wl_grid, d_modulus, opacities):
    ext_lumins = calc_dusty_transm(base_temp_lums, extinc_arr) * opacities
    zshift_wls = (1.+z)*wl_grid
    #f_ab = calc_fab(filts, zshift_wls, calc_dusty_transm(base_temp_lums, extinc_arr), d_modulus)
    #scale = calc_scale_arrs(calc_fab(filts, zshift_wls, calc_dusty_transm(base_temp_lums, extinc_arr), d_modulus), galax_fab, galax_fab_err)
    #scaled_lumins = ext_lumins*scale
    return calc_fab(filts,\
                    zshift_wls,\
                    ext_lumins*calc_scale_arrs(calc_fab(filts, zshift_wls, ext_lumins, d_modulus), galax_fab, galax_fab_err),\
                    d_modulus)

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
    #avmago = jnp.sum(arr_o*arr_t)
    #avmagt = jnp.sum(jnp.power(arr_t, 2.))
    return jnp.sum(arr_o*arr_t)/jnp.sum(jnp.power(arr_t, 2.))
    #else:
    #    _scale = 1.
    