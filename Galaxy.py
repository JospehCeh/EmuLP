#!/bin/env python3

from functools import partial
import numpy as np
import jax.numpy as jnp
from jax import jit, vmap, debug
import jax_cosmo as jc
from EmuLP import Filter, Template, Cosmology
from copy import deepcopy
from collections import namedtuple

Observation = namedtuple('Observation', ['num', 'AB_fluxes', 'AB_f_errors', 'valid_filters', 'z_spec'])

def load_galaxy(photometry, ismag):
    assert len(photometry)%2 == 0, "Missing data in observations : check that magnitudes/fluxes and errors are available\n and listed as M (or F), error, M (or F), error, etc."
    _phot = jnp.array( [ photometry[2*i] for i in range(len(photometry)//2) ] )
    _phot_errs = jnp.array( [ photometry[2*i+1] for i in range(len(photometry)//2) ] )

    if ismag:
        f_ab = jnp.power(10, -0.4*(_phot+48.6)) # conversion to flux.
        f_ab_err = _phot*_phot_errs/1.086 # As in LEPHARE - to be checked.
        filters_to_use = jnp.isfinite(f_ab)
        #magnitudes = _phot
        #mag_errors = _phot_errs
    else :
        f_ab = _phot
        f_ab_err = _phot_errs
        filters_to_use = (f_ab > 0.)
        #magnitudes = -2.5*jnp.log10(_phot)-48.6 # conversion to AB-magnitudes.
        #mag_errors = 1.086*_phot_errs/_phot # As in LEPHARE - to be checked.
    return f_ab, f_ab_err, filters_to_use

@jit
@vmap
def chi_term(obs, ref, err):
    return jnp.power((obs-ref)/err, 2.)

@jit
def z_prior_val(gal_fab, zp, base_temp_lums, extinc_arr, wl_grid, id_i_band=4):
    i_mag = -2.5*jnp.log10(gal_fab[id_i_band])-48.6
    nuvk = Template.calc_nuvk(base_temp_lums, extinc_arr, wl_grid)
    alpt0, zot, kt, pcal, ktf_m, ft_m = Cosmology.prior_alpt0(nuvk),\
                                        Cosmology.prior_zot(nuvk),\
                                        Cosmology.prior_kt(nuvk),\
                                        Cosmology.prior_pcal(nuvk),\
                                        Cosmology.prior_ktf(nuvk),\
                                        Cosmology.prior_ft(nuvk)
    val_prior = Cosmology.nz_prior_core(zp, i_mag, alpt0, zot, kt, pcal, ktf_m, ft_m)
    return val_prior

@jit
def z_ebv_prior_val(gal_fab, zp, base_temp_lums, extinc_arr, ext_law, ebv_val, wl_grid, id_i_band=4):
    i_mag = -2.5*jnp.log10(gal_fab[id_i_band])-48.6
    nuvk = Template.calc_nuvk(base_temp_lums, extinc_arr, wl_grid)
    alpt0, zot, kt, pcal, ktf_m, ft_m, mod = Cosmology.prior_alpt0(nuvk),\
                                                Cosmology.prior_zot(nuvk),\
                                                Cosmology.prior_kt(nuvk),\
                                                Cosmology.prior_pcal(nuvk),\
                                                Cosmology.prior_ktf(nuvk),\
                                                Cosmology.prior_ft(nuvk),\
                                                Cosmology.prior_mod(nuvk)
    val_prior = Cosmology.nz_prior_core(zp, i_mag, alpt0, zot, kt, pcal, ktf_m, ft_m) * Cosmology.ebv_prior(ebv_val, mod, ext_law)
    return val_prior

def noV_est_chi2(gal_fab, gal_fab_err, zp, base_temp_lums, extinc_arr, filters, cosmo, wl_grid, opacities):
    _selOpa = (wl_grid<1300.)
    interp_opac = jnp.interp(wl_grid, wl_grid[_selOpa], opacities, left=1., right=1., period=None)
    temp_fab = Template.noJit_make_scaled_template(base_temp_lums, filters, extinc_arr, zp, cosmo, gal_fab, gal_fab_err, wl_grid, interp_opac)
    _terms = chi_term(gal_fab, temp_fab, gal_fab_err)
    chi2 = jnp.sum(_terms)/len(_terms)
    return chi2

#@partial(jit, static_argnums=6)
@partial(vmap, in_axes=(None, None, None, 0, None, None, None, None, None, None))
@partial(vmap, in_axes=(None, None, None, None, 0, None, None, None, None, None))
@partial(vmap, in_axes=(None, None, 0, None, None, None, None, None, 0, None))
def est_chi2_prior(gal_fab, gal_fab_err, zp, base_temp_lums, extinc_arr, filters, cosmo, wl_grid, opacities, prior_band):
    #dist_mod = Cosmology.distMod(cosmo, zp)
    #prior_zp = z_prior_val(gal_fab, zp, base_temp_lums, extinc_arr, wl_grid)
    #zshift_wls = (1.+zp)*wl_grid
    #temp_fab = Template.make_scaled_template(base_temp_lums, filters, extinc_arr, gal_fab, gal_fab_err, (1.+zp)*wl_grid, Cosmology.distMod(cosmo, zp))
    _terms = chi_term(gal_fab,\
                      Template.make_scaled_template(base_temp_lums, filters, extinc_arr, gal_fab, gal_fab_err,\
                                                    zp, wl_grid, Cosmology.distMod(cosmo, zp), opacities),\
                      gal_fab_err)
    return jnp.sum(_terms)/len(_terms) - 2*jnp.log(z_prior_val(gal_fab, zp, base_temp_lums, extinc_arr, wl_grid, prior_band))

@partial(vmap, in_axes=(None, None, None, 0, None, None, None, None, None, None, None, None))
@partial(vmap, in_axes=(None, None, None, None, 0, 0, 0, None, None, None, None, None))
@partial(vmap, in_axes=(None, None, 0, None, None, None, None, None, None, None, 0, None))
def est_chi2_prior_ebv(gal_fab, gal_fab_err, zp, base_temp_lums, extinc_arr, ext_law, ebv_val, filters, cosmo, wl_grid, opacities, prior_band):
    #dist_mod = Cosmology.distMod(cosmo, zp)
    #prior_zp = z_prior_val(gal_fab, zp, base_temp_lums, extinc_arr, wl_grid)
    #zshift_wls = (1.+zp)*wl_grid
    #temp_fab = Template.make_scaled_template(base_temp_lums, filters, extinc_arr, gal_fab, gal_fab_err, (1.+zp)*wl_grid, Cosmology.distMod(cosmo, zp))
    _terms = chi_term(gal_fab,\
                      Template.make_scaled_template(base_temp_lums, filters, extinc_arr, gal_fab, gal_fab_err,\
                                                    zp, wl_grid, Cosmology.distMod(cosmo, zp), opacities),\
                      gal_fab_err)
    return jnp.sum(_terms)/len(_terms) - 2*jnp.log(z_ebv_prior_val(gal_fab, zp, base_temp_lums, extinc_arr, ext_law, ebv_val, wl_grid, prior_band))

#@partial(jit, static_argnums=6)
@partial(vmap, in_axes=(None, None, None, 0, None, None, None, None, None))
@partial(vmap, in_axes=(None, None, None, None, 0, None, None, None, None))
@partial(vmap, in_axes=(None, None, 0, None, None, None, None, None, 0))
def est_chi2(gal_fab, gal_fab_err, zp, base_temp_lums, extinc_arr, filters, cosmo, wl_grid, opacities):
    #dist_mod = Cosmology.distMod(cosmo, zp)
    #zshift_wls = (1.+zp)*wl_grid
    #temp_fab = Template.make_scaled_template(base_temp_lums, filters, extinc_arr, gal_fab, gal_fab_err, (1.+zp)*wl_grid, Cosmology.distMod(cosmo, zp))
    _terms = chi_term(gal_fab,\
                      Template.make_scaled_template(base_temp_lums, filters, extinc_arr, gal_fab, gal_fab_err,\
                                                    zp, wl_grid, Cosmology.distMod(cosmo, zp), opacities),\
                      gal_fab_err)
    if len(filters)<7 : debug.print("chi in bands = {t}", t=_terms)
    return jnp.sum(_terms)/len(_terms)

#@partial(jit, static_argnums=6)
@partial(vmap, in_axes=(None, None, None, 0, None, None, None, None, None, None))
@partial(vmap, in_axes=(None, None, None, None, 0, None, None, None, None, None))
@partial(vmap, in_axes=(None, None, 0, None, None, None, None, None, 0, None))
def est_chi2_prior_jaxcosmo(gal_fab, gal_fab_err, zphot, base_temp_lums, extinc_arr, filters, j_cosmo, wl_grid, opacities, prior_band):
    #dist_mod = 5.*jnp.log10(jnp.power((1.+zphot), 2)*jc.background.angular_diameter_distance(j_cosmo, jc.utils.z2a(zphot))/j_cosmo.h * 1.0e6) - 5.0 #Cosmology.calc_distMod(j_cosmo, zphot)
    #prior_zp = z_prior_val(gal_fab, zphot, base_temp_lums, extinc_arr, wl_grid)
    #zshift_wls = (1.+zphot)*wl_grid
    #temp_fab = Template.make_scaled_template(base_temp_lums, filters, extinc_arr, gal_fab, gal_fab_err, (1.+zphot)*wl_grid, Cosmology.calc_distMod(j_cosmo, zphot))
    _terms = chi_term(gal_fab,\
                      Template.make_scaled_template(base_temp_lums, filters, extinc_arr, gal_fab, gal_fab_err,\
                                                    zphot, wl_grid, Cosmology.calc_distMod(j_cosmo, zp), opacities),\
                      gal_fab_err)
    if len(filters)<7 : debug.print("chi in bands = {t}", t=_terms)
    #chi2 = jnp.sum(_terms)/len(_terms) - 2*jnp.log(z_prior_val(gal_fab, zphot, base_temp_lums, extinc_arr, wl_grid))
    return jnp.sum(_terms)/len(_terms) - 2*jnp.log(z_prior_val(gal_fab, zphot, base_temp_lums, extinc_arr, wl_grid, prior_band))

@partial(vmap, in_axes=(None, None, None, 0, None, None, None, None, None, None, None))
@partial(vmap, in_axes=(None, None, None, None, 0, 0, None, None, None, None, None))
@partial(vmap, in_axes=(None, None, 0, None, None, None, None, None, None, 0, None))
def est_chi2_prior_ebv_jaxcosmo(gal_fab, gal_fab_err, zphot, base_temp_lums, extinc_arr, ebv_val, filters, j_cosmo, wl_grid, opacities, prior_band):
    #dist_mod = 5.*jnp.log10(jnp.power((1.+zphot), 2)*jc.background.angular_diameter_distance(j_cosmo, jc.utils.z2a(zphot))/j_cosmo.h * 1.0e6) - 5.0 #Cosmology.calc_distMod(j_cosmo, zphot)
    #prior_zp = z_prior_val(gal_fab, zphot, base_temp_lums, extinc_arr, wl_grid)
    #zshift_wls = (1.+zphot)*wl_grid
    #temp_fab = Template.make_scaled_template(base_temp_lums, filters, extinc_arr, gal_fab, gal_fab_err, (1.+zphot)*wl_grid, Cosmology.calc_distMod(j_cosmo, zphot))
    _terms = chi_term(gal_fab,\
                      Template.make_scaled_template(base_temp_lums, filters, extinc_arr, gal_fab, gal_fab_err,\
                                                    zphot, wl_grid, Cosmology.calc_distMod(j_cosmo, zp), opacities),\
                      gal_fab_err)
    if len(filters)<7 : debug.print("chi in bands = {t}", t=_terms)
    #chi2 = jnp.sum(_terms)/len(_terms) - 2*jnp.log(z_prior_val(gal_fab, zphot, base_temp_lums, extinc_arr, wl_grid))
    return jnp.sum(_terms)/len(_terms) - 2*jnp.log(z_ebv_prior_val(gal_fab, zphot, base_temp_lums, extinc_arr, ebv_val, wl_grid, prior_band))

#@partial(jit, static_argnums=6)
@partial(vmap, in_axes=(None, None, None, 0, None, None, None, None, None))
@partial(vmap, in_axes=(None, None, None, None, 0, None, None, None, None))
@partial(vmap, in_axes=(None, None, 0, None, None, None, None, None, 0))
def est_chi2_jaxcosmo(gal_fab, gal_fab_err, zphot, base_temp_lums, extinc_arr, filters, j_cosmo, wl_grid, opacities):
    #dist_mod = 5.*jnp.log10(jnp.power((1.+zphot), 2)*jc.background.angular_diameter_distance(j_cosmo, jc.utils.z2a(zphot))/j_cosmo.h * 1.0e6) - 5.0 #Cosmology.calc_distMod(j_cosmo, zphot)
    #zshift_wls = (1.+zphot)*wl_grid
    #temp_fab = Template.make_scaled_template(base_temp_lums, filters, extinc_arr, gal_fab, gal_fab_err, zshift_wls = (1.+zphot)*wl_grid, Cosmology.calc_distMod(cosmo, zphot))
    _terms = chi_term(gal_fab,\
                      Template.make_scaled_template(base_temp_lums, filters, extinc_arr, gal_fab, gal_fab_err,\
                                                    zphot, wl_grid, Cosmology.calc_distMod(j_cosmo, zp), opacities),\
                      gal_fab_err)
    if len(filters)<7 : debug.print("chi in bands = {t}", t=_terms)
    #chi2 = jnp.sum(_terms)/len(_terms)
    return jnp.sum(_terms)/len(_terms)