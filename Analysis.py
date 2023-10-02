#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#  Analysis.py
#  
#  Copyright 2023  <joseph@wl-chevalier>
#  
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#  
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301, USA.
#  
#  


from EmuLP import Cosmology, Filter, Galaxy, Estimator, Extinction, Template

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import jax_cosmo as jc
import jax.numpy as jnp
from jax import vmap, jit
import json, pickle
import os,sys,copy
from tqdm import tqdm
from collections import namedtuple

from scipy.interpolate import interp1d
from jax_cosmo.scipy.interpolate import InterpolatedUnivariateSpline as j_spline

Cosmo = namedtuple('Cosmo', ['h0', 'om0', 'l0', 'omt'])
sedpyFilter = namedtuple('sedpyFilter', ['name', 'wavelengths', 'transmission'])
BaseTemplate = namedtuple('BaseTemplate', ['name', 'flux'])
Observation = namedtuple('Observation', ['num', 'AB_fluxes', 'AB_f_errors', 'z_spec'])
DustLaw = namedtuple('DustLaw', ['name', 'EBV', 'transmission'])

#conf_json = 'EmuLP/COSMOS2020-with-FORS2-HSC_only-jax-CC-togglePriorTrue-opa.json' # attention à la localisation du fichier !

def load_data_for_analysis(conf_json):
    with open(conf_json, "r") as inpfile:
        inputs = json.load(inpfile)

    #cosmo = Cosmology.make_jcosmo(inputs['Cosmology']['h0'])
    cosmo = Cosmo(inputs['Cosmology']['h0'], inputs['Cosmology']['om0'], inputs['Cosmology']['l0'],\
                  inputs['Cosmology']['om0']+inputs['Cosmology']['l0'])

    z_grid = jnp.arange(inputs['Z_GRID']['z_min'],\
                        inputs['Z_GRID']['z_max']+inputs['Z_GRID']['z_step'],\
                        inputs['Z_GRID']['z_step'])

    fine_z_grid = jnp.arange(inputs['Z_GRID']['z_min'],\
                             inputs['Z_GRID']['z_max']+min(0.01,inputs['Z_GRID']['z_step']),\
                             min(0.01,inputs['Z_GRID']['z_step']))

    wl_grid = jnp.arange(inputs['WL_GRID']['lambda_min'],\
                         inputs['WL_GRID']['lambda_max']+inputs['WL_GRID']['lambda_step'],\
                         inputs['WL_GRID']['lambda_step'])

    print("Loading filters :")
    filters_dict = inputs['Filters']
    filters_arr = tuple( sedpyFilter(*Filter.load_filt(filters_dict[ident]["name"],\
                                                       filters_dict[ident]["path"],\
                                                       filters_dict[ident]["transmission"]\
                                                      )\
                                    )
                        for ident in tqdm(filters_dict) )
    N_FILT = len(filters_arr)



    filters_jarr = tuple( Filter.sedpyFilter(*Filter.load_filt(int(ident),\
                                                               filters_dict[ident]["path"],\
                                                               filters_dict[ident]["transmission"]\
                                                              )\
                                            )
                         for ident in filters_dict )

    print("Building templates :")
    templates_dict = inputs['Templates']
    baseTemp_arr = tuple( BaseTemplate(*Template.make_base_template(templates_dict[ident]["name"],\
                                                                    templates_dict[ident]["path"],\
                                                                    wl_grid
                                                                   )\
                                      )
                         for ident in tqdm(templates_dict) )

    #baseFluxes_arr = jnp.row_stack((bt.flux for bt in baseTemp_arr))

    print("Generating dust attenuations laws :")
    extlaws_dict = inputs['Extinctions']
    dust_arr = []
    for ident in tqdm(extlaws_dict):
        dust_arr.extend([ DustLaw(extlaws_dict[ident]['name'],\
                                     ebv,\
                                     Extinction.load_extinc(extlaws_dict[ident]['path'],\
                                                            ebv,\
                                                            wl_grid)\
                                    )\
                            for ebv in tqdm(inputs['e_BV'])\
                           ])

    #extlaws_arr = jnp.row_stack( (dustlaw.transmission for dustlaw in dust_arr) )

    print("Loading IGM attenuations :")
    opa_path = os.path.abspath(inputs['Opacity'])
    _selOpa = (wl_grid < 1300.)
    wls_opa = wl_grid[_selOpa]
    opa_zgrid, opacity_grid = Extinction.load_opacity(opa_path, wls_opa)

    print("Loading observations :")
    data_path = os.path.abspath(inputs['Dataset']['path'])
    data_ismag = (inputs['Dataset']['type'].lower() == 'm')

    data_file_arr = np.loadtxt(data_path)
    _obs_arr = []

    for i in tqdm(range(data_file_arr.shape[0])):
        try:
            assert (len(data_file_arr[i,:]) == 1+2*N_FILT) or (len(data_file_arr[i,:]) == 1+2*N_FILT+1), f"At least one filter is missing in datapoint {data_file_arr[i,0]} : length is {len(data_file_arr[i,:])}, {1+2*N_FILT} values expected.\nDatapoint removed from dataset."
            #print(int(data_file_arr[i, 0]))
            if (len(data_file_arr[i,:]) == 1+2*N_FILT+1):
                observ = Observation(int(data_file_arr[i, 0]),\
                                     *Galaxy.load_galaxy(data_file_arr[i, 1:2*N_FILT+1],\
                                                         data_ismag),\
                                     data_file_arr[i, 2*N_FILT+1]\
                                    )
            else:
                observ = Observation(int(data_file_arr[i, 0]),\
                                     *Galaxy.load_galaxy(data_file_arr[i, 1:2*N_FILT+1],\
                                                         data_ismag),\
                                     None\
                                    )
            #print(observ.num)
            _obs_arr.extend([observ])
        except AssertionError:
            pass
        
    return cosmo, z_grid, fine_zgrid, wl_grid, filters_arr, filters_jarr, baseTemp_arr, dust_arr, wls_opa, opa_zgrid, opacity_grid, _obs_arr

def load_data_for_run(inputs):
    #cosmo = Cosmology.make_jcosmo(inputs['Cosmology']['h0'])
    if inputs['Cosmology']['jax-cosmo']:
        cosmo = Cosmology.make_jcosmo(inputs['Cosmology']['h0'])
    else:
        cosmo = Cosmology.Cosmo(inputs['Cosmology']['h0'], inputs['Cosmology']['om0'], inputs['Cosmology']['l0'],\
                                inputs['Cosmology']['om0']+inputs['Cosmology']['l0'])
    
    z_grid = jnp.arange(inputs['Z_GRID']['z_min'],\
                        inputs['Z_GRID']['z_max']+inputs['Z_GRID']['z_step'],\
                        inputs['Z_GRID']['z_step'])
    wl_grid = jnp.arange(inputs['WL_GRID']['lambda_min'],\
                         inputs['WL_GRID']['lambda_max']+inputs['WL_GRID']['lambda_step'],\
                         inputs['WL_GRID']['lambda_step'])
    
    filters_dict = inputs['Filters']
    print("Loading filters :")
    filters_arr = tuple( Filter.sedpyFilter(*Filter.load_filt(int(ident),\
                                                              filters_dict[ident]["path"],\
                                                              filters_dict[ident]["transmission"]\
                                                             )\
                                           )
                        for ident in tqdm(filters_dict) )
    N_FILT = len(filters_arr)
    #print(f"DEBUG: filters = {filters_arr}")
    
    print("Building templates :")
    templates_dict = inputs['Templates']
    baseTemp_arr = tuple( Template.BaseTemplate(*Template.make_base_template(templates_dict[ident]["name"],\
                                                                             templates_dict[ident]["path"],\
                                                                             wl_grid
                                                                            )\
                                               )
                         for ident in tqdm(templates_dict) )
    baseFluxes_arr = jnp.row_stack((bt.flux for bt in baseTemp_arr))
    
    print("Computing dust extinctions :")
    extlaws_dict = inputs['Extinctions']
    dust_arr = []
    for ident in tqdm(extlaws_dict):
        dust_arr.extend([ Extinction.DustLaw(extlaws_dict[ident]['name'],\
                                             ebv,\
                                             Extinction.load_extinc(extlaws_dict[ident]['path'],\
                                                                    ebv,\
                                                                    wl_grid)\
                                            )\
                            for ebv in tqdm(inputs['e_BV'])\
                           ])
    extlaws_arr = jnp.row_stack( (dustlaw.transmission for dustlaw in dust_arr) )
    #print(f"DEBUG: extinctions have shape {extlaws_arr.shape}, expected (nb(ebv)*nb(laws)={len(inputs['e_BV'])*len(extlaws_dict)}, len(wl_grid)={len(wl_grid)})")
    
    print("Loading IGM attenuations :")
    opa_path = os.path.abspath(inputs['Opacity'])
    _selOpa = (wl_grid < 1300.)
    wls_opa = wl_grid[_selOpa]
    opa_zgrid, opacity_grid = Extinction.load_opacity(opa_path, wls_opa)
    extrap_ones = jnp.ones((len(z_grid), len(wl_grid)-len(wls_opa)))
    print("Interpolating IGM attenuations :")
    interpolated_opacities = Extinction.opacity_at_z(z_grid, opa_zgrid, opacity_grid)
    interpolated_opacities = jnp.concatenate((interpolated_opacities, extrap_ones), axis=1)
    
    print('Loading observations :')
    data_path = os.path.abspath(inputs['Dataset']['path'])
    data_ismag = (inputs['Dataset']['type'].lower() == 'm')
    
    data_file_arr = loadtxt(data_path)
    obs_arr = []
    
    for i in tqdm(range(data_file_arr.shape[0])):
        try:
            assert (len(data_file_arr[i,:]) == 1+2*N_FILT) or (len(data_file_arr[i,:]) == 1+2*N_FILT+1), f"At least one filter is missing in datapoint {data_file_arr[i,0]} : length is {len(data_file_arr[i,:])}, {1+2*N_FILT} values expected.\nDatapoint removed from dataset."
            #print(int(data_file_arr[i, 0]))
            if (len(data_file_arr[i,:]) == 1+2*N_FILT+1):
                observ = Galaxy.Observation(int(data_file_arr[i, 0]),\
                                            *Galaxy.load_galaxy(data_file_arr[i, 1:2*N_FILT+1],\
                                                                data_ismag),\
                                            data_file_arr[i, 2*N_FILT+1]\
                                           )
            else:
                observ = Galaxy.Observation(int(data_file_arr[i, 0]),\
                                            *Galaxy.load_galaxy(data_file_arr[i, 1:2*N_FILT+1],\
                                                                data_ismag),\
                                            None\
                                           )
            #print(observ.num)
            obs_arr.extend([observ])
        except AssertionError:
            pass
        
    return cosmo, z_grid, wl_grid, filters_arr, filters_jarr, baseTemp_arr, baseFluxes_arr, extlaws_dict, dust_arr, extlaws_arr, interpolated_opacities, obs_arr


def results_in_dataframe(conf_json, observations, filters):
    with open(conf_json, "r") as inpfile:
        inputs = json.load(inpfile)
    df_res = pd.read_pickle(f"{inputs['run name']}_results.pkl")
    for obs in tqdm(observations):
    for i,filt in enumerate(filters):
        if obs.num in df_res.index:
            df_res.loc[obs.num, f"MagAB({filt.name})"] = -2.5*jnp.log10(obs.AB_fluxes[i])-48.6
            df_res.loc[obs.num, f"err_MagAB({filt.name})"] = 1.086*obs.AB_f_errors[i]/obs.AB_fluxes[i]
    df_res['bias'] = df_res['Photometric redshift']-df_res['True redshift']
    df_res['std'] = df_res['bias']/(1.+df_res['True redshift'])
    df_res['Outlier'] = np.abs(df_res['std'])>0.15
    df_res['G-R'] = df_res['MagAB(hsc_gHSC)']-df_res['MagAB(hsc_rHSC)']
    df_res['I-Z'] = df_res['MagAB(hsc_iHSC)']-df_res['MagAB(hsc_zHSC)']
    df_res['redness'] = df_res['G-R']/df_res['I-Z']
    outl_rate = 100.0*len(df_res[df_res['Outlier']])/len(df_res)
    
    return df_res, outl_rate

def probability_distrib(chi2_array):
    # Compute the probability values
    probs_array = jnp.exp(-0.5*chi2_array)
    
    # Integrate successively:
    ## Over models
    _int_mods = jnp.trapz(probs_array, x=jnp.arange(1, 1+len(baseTemp_arr)), axis=0)

    _sub_ints = jnp.split(_int_mods, len(inputs['Extinctions']), axis=0)
    sub_ints_ebv_z = []
    for sub_arr in _sub_ints:
        ## Integration over E(B-V)
        _int_ebv = jnp.trapz(sub_arr, x=jnp.array(inputs['e_BV']), axis=0)

        ## Over z
        _int_z = jnp.trapz(_int_ebv, x=z_grid, axis=0)

        sub_ints_ebv_z.append(_int_z)

    ## Over laws
    _int_laws = jnp.trapz(jnp.array(sub_ints_ebv_z), x=jnp.arange(1, 1+len(sub_ints_ebv_z)), axis=0)
    
    return probs_array / _int_laws, _int_laws

def prob_mod(probs_array):
    # Integrate successively:
    _sub_ints = jnp.split(probs_array, len(inputs['Extinctions']), axis=1)
    sub_ints_ebv_z = []
    for sub_arr in _sub_ints:
        ## Integration over E(B-V)
        _int_ebv = jnp.trapz(sub_arr, x=jnp.array(inputs['e_BV']), axis=1)
        ## Over z
        _int_z = jnp.trapz(_int_ebv, x=z_grid, axis=1)
        sub_ints_ebv_z.append(_int_z)

    ## Over laws
    sub_ints_ebv_z_arr = jnp.array(sub_ints_ebv_z)
    _int_laws = jnp.trapz(sub_ints_ebv_z_arr, x=jnp.arange(1, 1+len(sub_ints_ebv_z)), axis=0)
    return _int_laws

def prob_ebv(probs_array):
    # Integrate successively:
    ## Over models
    _int_mods = jnp.trapz(probs_array, x=jnp.arange(1, 1+len(baseTemp_arr)), axis=0)
    
    _sub_ints = jnp.split(_int_mods, len(inputs['Extinctions']), axis=0)
    sub_ints_z = []
    for sub_arr in _sub_ints:
        ## Over z
        _int_z = jnp.trapz(sub_arr, x=z_grid, axis=1)
        sub_ints_z.append(_int_z)

    ## Over laws
    sub_ints_z_arr = jnp.array(sub_ints_z)
    _int_laws = jnp.trapz(sub_ints_z_arr, x=jnp.arange(1, 1+len(sub_ints_z)), axis=0)
    return _int_laws

def prob_z(probs_array):
    # Integrate successively:
    ## Over models
    _int_mods = jnp.trapz(probs_array, x=jnp.arange(1, 1+len(baseTemp_arr)), axis=0)
    
    _sub_ints = jnp.split(_int_mods, len(inputs['Extinctions']), axis=0)
    sub_ints_ebv = []
    for sub_arr in _sub_ints:
        ## Integration over E(B-V)
        _int_ebv = jnp.trapz(sub_arr, x=jnp.array(inputs['e_BV']), axis=0)
        sub_ints_ebv.append(_int_ebv)

    ## Over laws
    sub_ints_ebv_arr = jnp.array(sub_ints_ebv)
    _int_laws = jnp.trapz(sub_ints_ebv_arr, x=jnp.arange(1, 1+len(sub_ints_ebv)), axis=0)
    return _int_laws

def prob_law(probs_array):
    # Integrate successively:
    ## Over models
    _int_mods = jnp.trapz(probs_array, x=jnp.arange(1, 1+len(baseTemp_arr)), axis=0)
    
    _sub_ints = jnp.split(_int_mods, len(inputs['Extinctions']), axis=0)
    sub_ints_ebv_z = []
    for sub_arr in _sub_ints:
        ## Integration over E(B-V)
        _int_ebv = jnp.trapz(sub_arr, x=jnp.array(inputs['e_BV']), axis=0)               
        ## Over z
        _int_z = jnp.trapz(_int_ebv, x=z_grid, axis=0)
        sub_ints_ebv_z.append(_int_z)
    sub_ints_z_arr = jnp.array(sub_ints_ebv_z)
    return sub_ints_z_arr

def evidence(probs_array, split_laws=False):
    # it is really just returning the array integrated over z
    if split_laws:
        # returned dimension will be nb of laws, nb of base templates, nb of E(B-V)
        _sub_ints = jnp.split(probs_array, len(inputs['Extinctions']), axis=1)
        sub_ints_z = []
        for sub_arr in _sub_ints:
            ## Over z
            _int_z = jnp.trapz(sub_arr, x=z_grid, axis=2)
            sub_ints_z.append(_int_z)
            res = jnp.array(sub_ints_z)
    else:
        # returned dimension will be nb of base templates, nb of laws * nb of E(B-V)
        res = jnp.trapz(probs_array, x=z_grid, axis=2)
    return res

def probs_at_fixed_z(probs_array, fixed_z, renormalize=True, prenormalize=False):
    # probs_array(n temp, n laws * n dust, len(z_grid)) -> probs_array(n temp, n laws * n dust)
    interpolated_array = jnp.zeros((probs_array.shape[0], probs_array.shape[1]))
    
    # Interpolate pdf at fixed_z
    for i in range(probs_array.shape[0]):
        for j in range(probs_array.shape[1]):
            _probs = probs_array[i,j,:]
            if prenormalize:
                _prenorm = jnp.trapz(_probs, x=z_grid, axis=0)
                _probs = _probs / _prenorm
            #f_interp = j_spline(z_grid, _probs, k=2)
            #_interp_pdf = f_interp(fixed_z)
            _interp_pdf = jnp.interp(fixed_z, z_grid, _probs)
            interpolated_array = interpolated_array.at[i,j].set(_interp_pdf)
    
    norm = 1.0
    if renormalize:
        # Integrate successively:
        ## Over models
        _int_mods = jnp.trapz(interpolated_array, x=jnp.arange(1, 1+len(baseTemp_arr)), axis=0)

        _sub_ints = jnp.split(_int_mods, len(inputs['Extinctions']), axis=0)
        sub_ints_ebv = []
        for sub_arr in _sub_ints:
            ## Integration over E(B-V)
            _int_ebv = jnp.trapz(sub_arr, x=jnp.array(inputs['e_BV']), axis=0)
            sub_ints_ebv.append(_int_ebv)
        ## Over laws
        norm = jnp.trapz(jnp.array(sub_ints_ebv), x=jnp.arange(1, 1+len(sub_ints_ebv)), axis=0)
    
    # return values array the same size as the number of based templates
    return interpolated_array / norm, norm

def p_template_at_fixed_z(probs_array, fixed_z):
    # probs_array(n temp, n laws * n dust, len(z_grid)) -> probs_array(n temp, n laws * n dust)
    interpolated_array, _norm = probs_at_fixed_z(probs_array, fixed_z, renormalize=True)
    
    # Split over dust extinction laws
    _sub_ints = jnp.split(interpolated_array, len(inputs['Extinctions']), axis=1)
    sub_ints_ebv = []
    for sub_arr in _sub_ints:
        ## Marginalize over E(B-V)
        _int_ebv = jnp.trapz(sub_arr, x=jnp.array(inputs['e_BV']), axis=1)
        sub_ints_ebv.append(_int_ebv)

    # Marginalize over extinction law
    sub_ints_ebv_arr = jnp.array(sub_ints_ebv)
    int_laws = jnp.trapz(sub_ints_ebv_arr, x=jnp.arange(1, 1+len(sub_ints_ebv)), axis=0)

    # return values array the same size as the number of based templates
    return int_laws

