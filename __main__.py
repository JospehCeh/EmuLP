#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#  __main__.py
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
from functools import partial
import jax.numpy as jnp
from jax import jit, vmap
from numpy import loadtxt
import numpy as np
import pandas as pd
import json
import os,sys,copy
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm

def main(args):
    if len(args) > 1:
        conf_json = args[1] # le premier argument de args est toujours `__main__.py`
    else :
        conf_json = 'EmuLP/defaults.json' # attention à la localisation du fichier !
    with open(conf_json, "r") as inpfile:
        inputs = json.load(inpfile)
    
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
    _obs_arr = []
    
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
            _obs_arr.extend([observ])
        except AssertionError:
            pass
    
    print('Photometric redshift estimation :')
    df_gal = pd.DataFrame()
    dict_of_results_dict = {}
    
    @partial(jit, static_argnums=(1,2))
    def estim_zp(observ, prior=True, jaxcosmo=False):
        if prior :
            if jaxcosmo:
                chi2_arr = Galaxy.est_chi2_prior_jaxcosmo(observ.AB_fluxes, observ.AB_f_errors,\
                                                          z_grid, baseFluxes_arr, extlaws_arr,\
                                                          filters_arr, cosmo, wl_grid,\
                                                          interpolated_opacities)
            else:
                chi2_arr = Galaxy.est_chi2_prior(observ.AB_fluxes, observ.AB_f_errors,\
                                                 z_grid, baseFluxes_arr, extlaws_arr,\
                                                 filters_arr, cosmo, wl_grid,\
                                                 interpolated_opacities)
        else:
            if jaxcosmo:
                chi2_arr = Galaxy.est_chi2_jaxcosmo(observ.AB_fluxes, observ.AB_f_errors,\
                                                    z_grid, baseFluxes_arr, extlaws_arr,\
                                                    filters_arr, cosmo, wl_grid,\
                                                    interpolated_opacities)
            else:
                chi2_arr = Galaxy.est_chi2(observ.AB_fluxes, observ.AB_f_errors,\
                                           z_grid, baseFluxes_arr, extlaws_arr,\
                                           filters_arr, cosmo, wl_grid,\
                                           interpolated_opacities)
        
        z_phot_loc = jnp.nanargmin(chi2_arr)
        return chi2_arr, z_phot_loc
        
    for i,observ in enumerate(tqdm(_obs_arr)):
        #print(observ.AB_fluxes)
        chi2_arr, z_phot_loc = estim_zp(observ, inputs['prior'], inputs['Cosmology']['jax-cosmo'])
        mod_num, ext_num, zphot_num = jnp.unravel_index(z_phot_loc,\
                                                        (len(baseTemp_arr),\
                                                         len(inputs['e_BV'])*len(extlaws_dict),\
                                                         len(z_grid)\
                                                        )\
                                                       )
        zphot, temp_id, law_id, ebv, chi2_val = z_grid[zphot_num],\
                                                baseTemp_arr[mod_num].name,\
                                                dust_arr[ext_num].name,\
                                                dust_arr[ext_num].EBV,\
                                                chi2_arr[(mod_num, ext_num, zphot_num)]
        #chi2_arr = Galaxy.est_chi2(observ.AB_fluxes, observ.AB_f_errors,\
        #                           z_grid, baseFluxes_arr, extlaws_arr,\
        #                           filters_arr, cosmo, wl_grid)
        #print(f"Shape of chi^2 array = {chi2_arr.shape} for observation {observ.num}, {i+1}/{int(len(_obs_arr))}.\n Expected shape is a mix of nb(templates)={int(len(baseTemp_arr))}, nb(ebv)*nb(laws)={len(inputs['e_BV'])*len(extlaws_dict)}, nb(redshift)={len(z_grid)}.")
        
        
        #z_phot_loc = jnp.nanargmin(chi2_arr) #, axis=2, keepdims=False)
        #zp_likelihood = jnp.exp(-0.5*chi2_arr)/jnp.sum(jnp.exp(-0.5*chi2_arr))
        #if not jnp.all(z_phot_loc==-1):
            # Point estimate = min(chi^2)
        #    mod_num, ext_num, zphot_num = jnp.unravel_index(z_phot_loc,\
        #                                                    (len(baseTemp_arr),\
        #                                                     len(inputs['e_BV'])*len(extlaws_dict),\
        #                                                     len(z_grid)\
        #                                                    )\
        #                                                   )
        #    zphot, temp_id, law_id, ebv, chi2_val = z_grid[zphot_num],\
        #                                            baseTemp_arr[mod_num].name,\
        #                                            dust_arr[ext_num].name,\
        #                                            dust_arr[ext_num].EBV,\
        #                                            chi2_arr[(mod_num, ext_num, zphot_num)]
        #print(f"For obs. {observ.num} with z_spec {observ.z_spec}, point estimates are :\n z_phot={zphot}, model num.={temp_id}, extinction law={law_id}, E(B-V)={ebv}, chi^2={chi2_val}")

        #for j,filt in enumerate(filters_arr):
        #    df_gal.loc[observ.num, f"Mag({filt.name})"] = -2.5*jnp.log10(observ.AB_fluxes[j])-48.6
        #    df_gal.loc[observ.num, f"MagErr({filt.name})"] = 1.086*observ.AB_f_errors[j]/observ.AB_fluxes[j]
        df_gal.loc[observ.num, "Photometric redshift"] = zphot
        df_gal.loc[observ.num, "True redshift"] = observ.z_spec
        df_gal.loc[observ.num, "Template SED"] = temp_id
        df_gal.loc[observ.num, "Extinction law"] = law_id
        df_gal.loc[observ.num, "E(B-V)"] = ebv
        df_gal.loc[observ.num, "Chi2"] = chi2_val

        # distributions
        #z_phot_loc = jnp.nanargmin(chi2_arr, axis=2)
        #tempId_arr = [ temp.name for temp in baseTemp_arr ]
        #extLaw_arr = [ ext.name for ext in dust_arr ]
        #eBV_arr = [ ext.EBV for ext in dust_arr ]
        #res_dict = {'zp': z_grid, 'chi2': chi2_arr, 'mod id': tempId_arr, 'ext law': extLaw_arr, 'eBV': eBV_arr, 'min_locs': z_phot_loc}
        dict_of_results_dict[observ.num] = chi2_arr
        
    if inputs["save results"]:
        df_gal.to_pickle(f"{inputs['run name']}_results.pkl")
        with open(f"{inputs['run name']}_results_dicts.pkl", 'wb') as handle:
            pickle.dump(dict_of_results_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    return 0

'''
def main(args):
    if len(args) > 1:
        conf_json = args[1] # le premier argument de args est toujours `__main__.py`
    else :
        conf_json = 'EmuLP/defaults.json' # attention à la localisation du fichier !
    with open(conf_json, "r") as inpfile:
        inputs = json.load(inpfile)
    lcdm = Cosmology.Cosmology(inputs['Cosmology']['h0'], inputs['Cosmology']['om0'], inputs['Cosmology']['l0'])
    print(lcdm)
    print(f"The Universe was {lcdm.time(1100):.0f} years old at CMB emission.")
    print(f"The Universe is {lcdm.time(0):.0f} years old now.")
    
    templates_dict = inputs['Templates']
    z_grid = jnp.arange(inputs['Z_GRID']['z_min'], inputs['Z_GRID']['z_max']+inputs['Z_GRID']['z_step'], inputs['Z_GRID']['z_step'])
    wl_grid = jnp.arange(inputs['WL_GRID']['lambda_min'], inputs['WL_GRID']['lambda_max']+inputs['WL_GRID']['lambda_step'], inputs['WL_GRID']['lambda_step'])
    extlaws_dict = inputs['Extinctions']
    #extlaws_arr = [ Extinction.Extinction(int(ident), extlaws_dict[ident]['path']) for ident in extlaws_dict.keys() ]
    
    filters_dict = inputs['Filters']
    filters_arr = [ Filter.Filter(int(ident), filters_dict[ident]["path"], filters_dict[ident]["transmission"]) for ident in filters_dict.keys() ]
    N_FILT = len(filters_arr)
    
    df_filters = Filter.to_df(filters_arr, wl_grid, inputs["save results"], f"{inputs['run name']}_filters")
    templates_arr = []
    
    for temp_ident in templates_dict:
        template = Template.Template(int(temp_ident), templates_dict[temp_ident]['path'])
        for extinction_id in extlaws_dict:
            for ebv in inputs['e_BV']:
                #_templ = copy.deepcopy(template)
                _templ = Template.apply_extinc(template, int(extinction_id), extlaws_dict[extinction_id]['path'], ebv)
                #for redshift in z_grid:
                #_templ_arr = _templ.to_redshift(z_grid, lcdm)
                _templ_arr = Template.to_redshift(_templ.toDict(), z_grid, lcdm.toDict())
                #template.normalize(1000., 10000.)
                _templ_arr = [ {'cosmo': _templ_arr['cosmo'],\
                                'd_angular': _templ_arr['d_angular'][_id],\
                                'd_luminosity': _templ_arr['d_luminosity'][_id],\
                                'd_metric': _templ_arr['d_metric'][_id],\
                                'd_modulus': _templ_arr['d_modulus'][_id],\
                                'ebv': _templ_arr['ebv'],\
                                'extinc_law': _templ_arr['extinc_law'],\
                                'id': _templ_arr['id'],\
                                'lumins': _templ_arr['lumins'][_id],\
                                'redshift': _templ_arr['redshift'][_id],\
                                'scaling_factor': _templ_arr['scaling_factor'],\
                                'wavelengths': _templ_arr['wavelengths'][_id]\
                               }\
                              for _id in range(len(z_grid)) ]
                for _id, _temp in enumerate(_templ_arr) :
                    _templ = Template.Template(-99, '', specdict=_temp)
                    _templ_arr[_id] = Template.fill_magnitudes(_templ, filters_arr)
                templates_arr.extend(_templ_arr)
                del _templ
        del template
    df_templates_prop, df_templates = Template.to_df(templates_arr, filters_arr, wl_grid,\
                                                     inputs["save results"], f"{inputs['run name']}_templates")
                    
    estim_method = inputs['Estimator']
    if estim_method.lower() == 'chi2':
        zp_estim = Estimator.Chi2(templates_arr) #Estimator.Chi2(estim_method, templates_arr)
    else:
        raise RuntimeError(f"Unimplemented estimator {estim_method}.\nPlease specify one of the following: chi2, <more to come>.")
    
    
    data_path = os.path.abspath(inputs['Dataset']['path'])
    data_ismag = (inputs['Dataset']['type'].lower() == 'm')
    
    data_file_arr = loadtxt(data_path)
    _galaxy_arr = []
    
    print("\n")
    for i in range(data_file_arr.shape[0]):
        try:
            assert (len(data_file_arr[i,:]) == 1+2*N_FILT) or (len(data_file_arr[i,:]) == 1+2*N_FILT+1), f"At least one filter is missing in datapoint {data_file_arr[i,0]} : length is {len(data_file_arr[i,:])}, {1+2*N_FILT} values expected.\nDatapoint removed from dataset."
            if (len(data_file_arr[i,:]) == 1+2*N_FILT+1):
                gal = Galaxy.Galaxy(data_file_arr[i, 0], data_file_arr[i, 1:2*N_FILT+1], data_ismag, zs=data_file_arr[i, 2*N_FILT+1])
            else:
                gal = Galaxy.Galaxy(data_file_arr[i, 0], data_file_arr[i, 1:2*N_FILT+1], data_ismag)
            
            _galaxy_arr.extend([gal])
        except AssertionError:
            pass

        #print(galaxy_arr[0])
        _progress = ((i+1)*100) // data_file_arr.shape[0]
        if _progress%2 == 1:
            #sys.stdout[-1] = (f"Estimation: {_progress}% done out of {data_file_arr.shape[0]} galaxies in dataset.")
            print(f"Data generation: {_progress}% done out of {data_file_arr.shape[0]} galaxies in dataset.", flush=True)
    all_ids = jnp.array([_galaxy_arr[k]["id"] for k in range(len(_galaxy_arr))])
    galaxy_arr = Galaxy.estimate_zp(all_ids, tuple(_galaxy_arr), zp_estim, tuple(filters_arr), 1)
    df_results = Galaxy.to_df(galaxy_arr, filters_arr, inputs["save results"], f"{inputs['run name']}_results")
    dict_of_results_dict = { _gal.id: _gal.results_dict for _gal in galaxy_arr }
    
    if inputs["save results"]:
        with open(f"{inputs['run name']}_results_dicts.pkl", 'wb') as handle:
            pickle.dump(dict_of_results_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    f,a = plt.subplots(1,1)
    df_results.plot.scatter("True redshift", "Photometric redshift", ax=a)
    f.show()
    
    return 0
'''

if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))
