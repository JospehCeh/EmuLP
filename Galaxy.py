#!/bin/env python3

from functools import partial
import numpy as np
import jax.numpy as jnp
from jax import jit, vmap, debug
import jax_cosmo as jc
from EmuLP import Filter, Template, Cosmology
from copy import deepcopy
import pandas as pd
import munch
from collections import namedtuple

Observation = namedtuple('Observation', ['num', 'AB_fluxes', 'AB_f_errors', 'z_spec'])

class Galaxy(munch.Munch):
    """
    Galaxy objects corresponding to each observation in the dataset.\n
    Photometry to be provided as a single array of length 2*n with n the number of filter in argument `photometry`\n
    and structured as MEME or FEFE (mag, error, etc. or flux, error, etc.).\n
    The type of photometry is to be specified as F or M in the argument `struct`.
    """
    
    def __init__(self, ident, photometry, ismag, zs=None, galdict=None):
        if galdict is None:
            super().__init__()
            assert len(photometry)%2 == 0, "Missing data in observations : check that magnitudes/fluxes and errors are available\n and listed as M (or F), error, M (or F), error, etc."
            _phot = jnp.array([])
            _phot_errs = jnp.array([])

            self.id = int(ident) # name or number, but as a string
            if zs is None:
                self.zs = None
                self.is_spectro = False
            else :
                self.zs = zs
                self.is_spectro = True

            for i in range(len(photometry)//2):
                _phot = jnp.append(_phot, photometry[2*i])
                _phot_errs = jnp.append(_phot_errs, photometry[2*i+1])

            if ismag:
                #self.obs_type = "AB-magnitudes"
                self.f_ab = jnp.power(10, -0.4*(_phot+48.6)) # conversion to flux.
                self.f_ab_err = _phot*_phot_errs/1.086 # As in LEPHARE - to be checked.
                self.magnitudes = _phot
                self.mag_errors = _phot_errs
            else :
                #self.obs_type = "fluxes"
                self.f_ab = _phot
                self.f_ab_err = _phot_errs
                self.magnitudes = -2.5*jnp.log10(_phot)-48.6 # conversion to AB-magnitudes.
                self.mag_errors = 1.086*_phot_errs/_phot # As in LEPHARE - to be checked.
                
            self.zp, self.template, self.ebv, self.extinction_law, self.chi2, self.results_dict = None, None, None, None, None, None
        else:
            super().__init__(galdict)
            if not "zp" in galdict.keys():
                self.zp, self.template, self.ebv, self.extinction_law, self.chi2, self.results_dict = None, None, None, None, None, None

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
    
    def __hash__(self):
        if self.zp is None:
            hashval = hash((self.id, self.is_spectro))
        else:
            hashval = hash((self.id, self.is_spectro, self.zp, self.template, self.extinction_law, self.ebv))
        return hashval
    
    def __eq__(self, other):
        _b1 = self.zp is None
        _b2 = other.zp is None
        if _b1 or _b2 : 
            res = (self.id==other.id) and (self.is_spectro==other.is_spectro) and (_b1==_b2)
        else :
            res = (self.id==other.id) and (self.is_spectro==other.is_spectro) and (self.zp==other.zp) and (self.template==other.template) and (self.extinction_law==other.extinction_law) and (self.ebv==other.ebv)
        return res
    
    def __ne__(self, other):
        return not self.__eq__(other)
    
    '''
    def estim_zp(self, estimator, filters):
        #self.zp, self.template, self.extinction_law, self.ebv, self.chi2 = estimator.estimate_zp(self, filters)
        self.zp, self.template, self.extinction_law, self.ebv, self.chi2, self.results_dict = estimator.estimate_zp(self, filters)
    '''
    
    ''' 
    @partial(vmap, in_axes=(0, 0, 0))
    def chi_term(obs, ref, err):
        return jnp.power((obs-ref)/err, 2.)
    '''
    
    @partial(vmap, in_axes=(None, None, 0))
    def chi2(self, filters, template):
        _scaled_temp, _scale = template.auto_scale(self, filters)

        _sel1 = jnp.isfinite(self.f_ab)
        _sel2 = jnp.isfinite(_scaled_temp.f_ab)
        _sel3 = jnp.isfinite(self.f_ab_err)
        _sel = [ (b1 and b2 and b3) for b1,b2,b3 in zip(_sel1, _sel2, _sel3) ]
        if len(_scaled_temp.f_ab[_sel]) > 1 :
            #_cols = -np.diff(self.f_ab[_sel])
            #_temp_cols = -np.diff(_scaled_temp.f_ab[_sel])
            #_errs = np.sqrt([x**2+y**2 for x,y in zip(self.f_ab_err[_sel][:-1], self.f_ab_err[_sel][1:])])
            
            '''
            _terms = jnp.array([ jnp.power((_fab - _scaled_temp_fab)/_err, 2.)\
                                for _fab,_err,_scaled_temp_fab in zip(self.f_ab[_sel],\
                                                                      self.f_ab_err[_sel],\
                                                                      _scaled_temp.f_ab[_sel])\
                              ])
            '''
            _terms = chi_term(self.f_ab[_sel], _scaled_temp.f_ab[_sel], self.f_ab_err[_sel])
            _chi2 = jnp.sum(_terms)/len(_terms)
        else :
            _chi2 = jnp.inf
        del _scaled_temp
        return _chi2 #, _scale
    
def to_arrays(gal):
    return gal.f_ab, gal.f_ab_err, gal.id

def load_galaxy(photometry, ismag):
    assert len(photometry)%2 == 0, "Missing data in observations : check that magnitudes/fluxes and errors are available\n and listed as M (or F), error, M (or F), error, etc."
    _phot = jnp.array( [ photometry[2*i] for i in range(len(photometry)//2) ] )
    _phot_errs = jnp.array( [ photometry[2*i+1] for i in range(len(photometry)//2) ] )

    if ismag:
        f_ab = jnp.power(10, -0.4*(_phot+48.6)) # conversion to flux.
        f_ab_err = _phot*_phot_errs/1.086 # As in LEPHARE - to be checked.
        #magnitudes = _phot
        #mag_errors = _phot_errs
    else :
        f_ab = _phot
        f_ab_err = _phot_errs
        #magnitudes = -2.5*jnp.log10(_phot)-48.6 # conversion to AB-magnitudes.
        #mag_errors = 1.086*_phot_errs/_phot # As in LEPHARE - to be checked.
    return f_ab, f_ab_err

@jit
@vmap
def chi_term(obs, ref, err):
    return jnp.power((obs-ref)/err, 2.)

'''
@partial(jit, static_argnums=(3,6))
@partial(vmap, in_axes=(None, None, None, None, 0, None, None, None))
@partial(vmap, in_axes=(None, None, 0, None, None, None, None, None))
def est_chi2(gal_fab, gal_fab_err, zp, temp_file, extinc_arr, filters, cosmo, wl_grid):
    temp_fab = Template.make_scaled_template(temp_file, filters, extinc_arr, zp, cosmo, gal_fab, gal_fab_err, wl_grid)
    #_sel1 = jnp.isfinite(gal_fab)
    #_sel2 = jnp.isfinite(temp_fab)
    #_sel3 = jnp.isfinite(gal_fab_err)
    #_sel = [ (b1 and b2 and b3) for b1,b2,b3 in zip(_sel1, _sel2, _sel3) ]
    #if len(temp_fab[_sel]) > 1 :
        #_cols = -np.diff(self.f_ab[_sel])
        #_temp_cols = -np.diff(_scaled_temp.f_ab[_sel])
        #_errs = np.sqrt([x**2+y**2 for x,y in zip(self.f_ab_err[_sel][:-1], self.f_ab_err[_sel][1:])])

    #_terms = jnp.array([ jnp.power((_fab - _scaled_temp_fab)/_err, 2.)\
    #                    for _fab,_err,_scaled_temp_fab in zip(self.f_ab[_sel],\
    #                                                          self.f_ab_err[_sel],\
    #                                                          _scaled_temp.f_ab[_sel])\
    #                  ])
    _terms = chi_term(gal_fab, temp_fab, gal_fab_err)
    chi2 = jnp.sum(_terms)/len(_terms)
    #else :
        #chi2 = jnp.inf
    return chi2
'''

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

def noV_est_chi2(gal_fab, gal_fab_err, zp, base_temp_lums, extinc_arr, filters, cosmo, wl_grid):
    temp_fab = Template.noJit_make_scaled_template(base_temp_lums, filters, extinc_arr, zp, cosmo, gal_fab, gal_fab_err, wl_grid)
    _terms = chi_term(gal_fab, temp_fab, gal_fab_err)
    chi2 = jnp.sum(_terms)/len(_terms)
    return chi2

#@partial(jit, static_argnums=6)
@partial(vmap, in_axes=(None, None, None, 0, None, None, None, None))
@partial(vmap, in_axes=(None, None, None, None, 0, None, None, None))
@partial(vmap, in_axes=(None, None, 0, None, None, None, None, None))
def est_chi2_prior(gal_fab, gal_fab_err, zp, base_temp_lums, extinc_arr, filters, cosmo, wl_grid):
    #dist_mod = Cosmology.distMod(cosmo, zp)
    #prior_zp = z_prior_val(gal_fab, zp, base_temp_lums, extinc_arr, wl_grid)
    #zshift_wls = (1.+zp)*wl_grid
    #temp_fab = Template.make_scaled_template(base_temp_lums, filters, extinc_arr, gal_fab, gal_fab_err, (1.+zp)*wl_grid, Cosmology.distMod(cosmo, zp))
    _terms = chi_term(gal_fab, Template.make_scaled_template(base_temp_lums, filters, extinc_arr, gal_fab, gal_fab_err, (1.+zp)*wl_grid, Cosmology.distMod(cosmo, zp)), gal_fab_err)
    return jnp.sum(_terms)/len(_terms) - 2*jnp.log(z_prior_val(gal_fab, zp, base_temp_lums, extinc_arr, wl_grid))

#@partial(jit, static_argnums=6)
@partial(vmap, in_axes=(None, None, None, 0, None, None, None, None))
@partial(vmap, in_axes=(None, None, None, None, 0, None, None, None))
@partial(vmap, in_axes=(None, None, 0, None, None, None, None, None))
def est_chi2(gal_fab, gal_fab_err, zp, base_temp_lums, extinc_arr, filters, cosmo, wl_grid):
    #dist_mod = Cosmology.distMod(cosmo, zp)
    #zshift_wls = (1.+zp)*wl_grid
    #temp_fab = Template.make_scaled_template(base_temp_lums, filters, extinc_arr, gal_fab, gal_fab_err, (1.+zp)*wl_grid, Cosmology.distMod(cosmo, zp))
    _terms = chi_term(gal_fab, Template.make_scaled_template(base_temp_lums, filters, extinc_arr, gal_fab, gal_fab_err, (1.+zp)*wl_grid, Cosmology.distMod(cosmo, zp)), gal_fab_err)
    return jnp.sum(_terms)/len(_terms)

#@partial(jit, static_argnums=6)
@partial(vmap, in_axes=(None, None, None, 0, None, None, None, None))
@partial(vmap, in_axes=(None, None, None, None, 0, None, None, None))
@partial(vmap, in_axes=(None, None, 0, None, None, None, None, None))
def est_chi2_prior_jaxcosmo(gal_fab, gal_fab_err, zphot, base_temp_lums, extinc_arr, filters, j_cosmo, wl_grid):
    #dist_mod = 5.*jnp.log10(jnp.power((1.+zphot), 2)*jc.background.angular_diameter_distance(j_cosmo, jc.utils.z2a(zphot))/j_cosmo.h * 1.0e6) - 5.0 #Cosmology.calc_distMod(j_cosmo, zphot)
    #prior_zp = z_prior_val(gal_fab, zphot, base_temp_lums, extinc_arr, wl_grid)
    #zshift_wls = (1.+zphot)*wl_grid
    #temp_fab = Template.make_scaled_template(base_temp_lums, filters, extinc_arr, gal_fab, gal_fab_err, (1.+zphot)*wl_grid, Cosmology.calc_distMod(j_cosmo, zphot))
    _terms = chi_term(gal_fab, Template.make_scaled_template(base_temp_lums, filters, extinc_arr, gal_fab, gal_fab_err, (1.+zphot)*wl_grid, Cosmology.calc_distMod(j_cosmo, zphot)), gal_fab_err)
    #chi2 = jnp.sum(_terms)/len(_terms) - 2*jnp.log(z_prior_val(gal_fab, zphot, base_temp_lums, extinc_arr, wl_grid))
    return jnp.sum(_terms)/len(_terms) - 2*jnp.log(z_prior_val(gal_fab, zphot, base_temp_lums, extinc_arr, wl_grid))

#@partial(jit, static_argnums=6)
@partial(vmap, in_axes=(None, None, None, 0, None, None, None, None))
@partial(vmap, in_axes=(None, None, None, None, 0, None, None, None))
@partial(vmap, in_axes=(None, None, 0, None, None, None, None, None))
def est_chi2_jaxcosmo(gal_fab, gal_fab_err, zphot, base_temp_lums, extinc_arr, filters, j_cosmo, wl_grid):
    #dist_mod = 5.*jnp.log10(jnp.power((1.+zphot), 2)*jc.background.angular_diameter_distance(j_cosmo, jc.utils.z2a(zphot))/j_cosmo.h * 1.0e6) - 5.0 #Cosmology.calc_distMod(j_cosmo, zphot)
    #zshift_wls = (1.+zphot)*wl_grid
    #temp_fab = Template.make_scaled_template(base_temp_lums, filters, extinc_arr, gal_fab, gal_fab_err, zshift_wls = (1.+zphot)*wl_grid, Cosmology.calc_distMod(cosmo, zphot))
    _terms = chi_term(gal_fab, Template.make_scaled_template(base_temp_lums, filters, extinc_arr, gal_fab, gal_fab_err, (1.+zphot)*wl_grid, Cosmology.calc_distMod(j_cosmo, zphot)), gal_fab_err)
    #chi2 = jnp.sum(_terms)/len(_terms)
    return jnp.sum(_terms)/len(_terms)

@partial(jit, static_argnums=(1,2,3,4))
@partial(vmap, in_axes=(0, None, None, None, None))
def estimate_zp(galid, galdictarr, estimator, filters, siz):
    #id_arr = jnp.array([ galdictarr[k]["id"] for k in range(len(galdictarr))])
    #_sel, = jnp.nonzero((id_arr == galid), size=siz, fill_value=0)
    bools = jnp.array([ galdictarr[k]["id"]==galid for k in range(len(galdictarr)) ])
    _sel, = jnp.nonzero(bools, size=siz, fill_value=0)
    _val = _sel[0].astype(int)
    
    @partial(jit, static_argnums=0)
    def select_gal_in_tup(index):
        return galdictarr[index]

    gal = select_gal_in_tup(_val)
    galaxy = estimator.estimate_zp(gal, filters)
    
    '''
    galaxy = Galaxy(None, None, None, zs=None, galdict=gal)
    galaxy.zp, galaxy.template, galaxy.extinction_law, galaxy.ebv, galaxy.chi2, galaxy.results_dict =\
    estimator.estimate_zp(galaxy, filters)
    '''
    
    return galaxy

def to_df(galaxies_arr, filters_arr, save=False, name=None):
    df_gal = pd.DataFrame(index=[_gal.id for _gal in galaxies_arr])
    for i,filt in enumerate(filters_arr):
        df_gal[f"Mag({filt.id})"] = [ _gal.magnitudes[i] for _gal in galaxies_arr ]
        df_gal[f"MagErr({filt.id})"] = [ _gal.mag_errors[i] for _gal in galaxies_arr ]
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

def return_updated_galaxy(galaxy, **kwargs):
    as_dict = galaxy.toDict()
    as_dict.update(kwargs)
    new_gal = Galaxy(None, None, None, zs=None, specdict=as_dict)
    return new_gal
