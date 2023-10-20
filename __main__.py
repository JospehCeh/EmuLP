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

from EmuLP import Cosmology, Filter, Galaxy, Estimator, Extinction, Template, Analysis
from functools import partial
import jax.numpy as jnp
from jax import jit, vmap, debug
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
    inputs = Analysis.json_to_inputs(conf_json)
    
    cosmo, z_grid, wl_grid,\
    filters_arr, named_filts,\
    baseTemp_arr, baseFluxes_arr,\
    extlaws_dict, ebv_vals, dust_arr, extlaws_arr, interpolated_opacities,\
    obs_arr = Analysis.load_data_for_run(inputs)
    
    
    _old_dir = os.getcwd()
    _path = os.path.abspath(__file__)
    _dname = os.path.dirname(_path)
    os.chdir(_dname)
    opa_path = os.path.abspath(inputs['Opacity'])
    os.chdir(_old_dir)
    
    _selOpa = (wl_grid < 1300.)
    wls_opa = wl_grid[_selOpa]
    opa_zgrid, opacity_grid = Extinction.load_opacity(opa_path, wls_opa)
    extrap_ones = jnp.ones((len(z_grid), len(wl_grid)-len(wls_opa)))
    
    print('Photometric redshift estimation :')
    df_gal = pd.DataFrame()
    dict_of_results_dict = {}
    
    #@partial(jit, static_argnums=(1,2))
    def estim_zp(observ, prior=True, jaxcosmo=False, id_prior_band=4):
        f = observ.AB_fluxes[observ.valid_filters]
        f_err = observ.AB_f_errors[observ.valid_filters]
        filts = []
        prior_band = -1
        for (_f, _b) in enumerate(observ.valid_filters):
            if _b:
                filts.append(filters_arr[_f])
                if _f <= id_prior_band:
                    prior_band+=1
        filts=tuple(filts)
        if (prior and observ.valid_filters[id_prior_band]) :
            if jaxcosmo:
                chi2_arr = Galaxy.est_chi2_prior_jaxcosmo(f,\
                                                          f_err,\
                                                          z_grid, baseFluxes_arr, extlaws_arr,\
                                                          filts, cosmo, wl_grid,\
                                                          interpolated_opacities, prior_band)
            else:
                chi2_arr = Galaxy.est_chi2_prior(f,\
                                                 f_err,\
                                                 z_grid, baseFluxes_arr, extlaws_arr,\
                                                 filts, cosmo, wl_grid,\
                                                 interpolated_opacities, prior_band)
        else:
            if jaxcosmo:
                chi2_arr = Galaxy.est_chi2_jaxcosmo(f,\
                                                    f_err,\
                                                    z_grid, baseFluxes_arr, extlaws_arr,\
                                                    filts, cosmo, wl_grid,\
                                                    interpolated_opacities)
            else:
                chi2_arr = Galaxy.est_chi2(f,\
                                           f_err,\
                                           z_grid, baseFluxes_arr, extlaws_arr,\
                                           filts, cosmo, wl_grid,\
                                           interpolated_opacities)
        
        z_phot_loc = jnp.nanargmin(chi2_arr)
        return chi2_arr, z_phot_loc
    
    empty_counts=0
    empty_list=[]
    for i,observ in enumerate(tqdm(obs_arr)):
        #print(observ.AB_fluxes)
        if jnp.any(observ.valid_filters):
            chi2_arr, z_phot_loc = estim_zp(observ, inputs['prior'], inputs['Cosmology']['jax-cosmo'])
            mod_num, ext_num, zphot_num = jnp.unravel_index(z_phot_loc,\
                                                            (len(baseTemp_arr),\
                                                             len(ebv_vals)*len(extlaws_dict),\
                                                             len(z_grid)\
                                                            )\
                                                           )
            
            zphot, temp_id, law_id, ebv, chi2_val = z_grid[zphot_num],\
                                                    baseTemp_arr[mod_num].name,\
                                                    dust_arr[ext_num].name,\
                                                    dust_arr[ext_num].EBV,\
                                                    chi2_arr[(mod_num, ext_num, zphot_num)]
            
            df_gal.loc[i, "Id"] = observ.num
            df_gal.loc[i, "Photometric redshift"] = zphot
            df_gal.loc[i, "True redshift"] = observ.z_spec
            df_gal.loc[i, "Template SED"] = temp_id
            df_gal.loc[i, "Extinction law"] = law_id
            df_gal.loc[i, "E(B-V)"] = ebv
            df_gal.loc[i, "Chi2"] = chi2_val

            probsarr, norm = Analysis.probability_distrib(chi2_arr, len(baseTemp_arr), len(extlaws_dict), ebv_vals, z_grid)
            
            while abs(1-norm)>1.0e-3 :
                chi2_arr = chi2_arr + 2*jnp.log(norm)
                probsarr, norm = Analysis.probability_distrib(chi2_arr, len(baseTemp_arr), len(extlaws_dict), ebv_vals, z_grid)

            NMOD = inputs['NMOD']
            evidence_ranked_mods = {}
            evidence_ranked_mods["Template SED"] = []
            evidence_ranked_mods["Dust law"] = []
            evidence_ranked_mods["E(B-V)"] = []
            evidence_ranked_mods["zp (mode)"] = []
            evidence_ranked_mods["average(z)"] = []
            evidence_ranked_mods["sigma(z)"] = []
            evidence_ranked_mods["median(z)"] = []
            evidence_ranked_mods["Odd ratio"] = []
            evidence_ranked_mods["Bias"] = []
            for f in named_filts:
                evidence_ranked_mods[f"M({f.name})"] = []

            mods_at_z_spec = {}
            mods_at_z_spec["Template SED"] = []
            mods_at_z_spec["Dust law"] = []
            mods_at_z_spec["E(B-V)"] = []
            mods_at_z_spec["zp (mode)"] = []
            mods_at_z_spec["average(z)"] = []
            mods_at_z_spec["sigma(z)"] = []
            mods_at_z_spec["median(z)"] = []
            mods_at_z_spec["Odd ratio"] = []
            mods_at_z_spec["Bias"] = []
            for f in named_filts:
                mods_at_z_spec[f"M({f.name})"] = []

            # Include evidence-derived properties
            evs_nosplit = Analysis.evidence(probsarr, len(extlaws_dict), z_grid, split_laws=False)
            sorted_evs_flat = jnp.argsort(evs_nosplit, axis=None)
            sorted_evs = [ jnp.unravel_index(idx, evs_nosplit.shape) for idx in sorted_evs_flat ]
            sorted_evs.reverse()
            n_temp, n_dust = sorted_evs[0]

            pz_at_ev = probsarr[n_temp, n_dust, :] / jnp.trapz(probsarr[n_temp, n_dust, :], x=z_grid)
            z_mean = jnp.trapz(z_grid*pz_at_ev, x=z_grid)
            z_std = jnp.trapz(pz_at_ev*jnp.power(z_grid-z_mean, 2), x=z_grid)
            try:
                z_mod = z_grid[jnp.nanargmax(pz_at_ev)]
            except ValueError:
                z_mod = jnp.nan
            df_gal.loc[i, "Highest evidence SED"] = baseTemp_arr[n_temp].name
            df_gal.loc[i, "Highest evidence dust law"] = dust_arr[n_dust].name
            df_gal.loc[i, "Highest evidence E(B-V)"] = dust_arr[n_dust].EBV
            df_gal.loc[i, "Highest evidence odd ratio"] = float(evs_nosplit[n_temp, n_dust] / evs_nosplit[mod_num, ext_num])
            df_gal.loc[i, "Highest evidence z_phot (mode)"] = z_mod
            df_gal.loc[i, "Highest evidence z_phot (mean)"] = z_mean
            df_gal.loc[i, "Highest evidence sigma(z)"] = z_std

            if inputs["Evidence analysis"]:
                # Include more evidence-derived properties
                for rank, (n_temp, n_dust) in enumerate(sorted_evs[:NMOD]):
                    evidence_ranked_mods["Template SED"].append(baseTemp_arr[n_temp].name)
                    evidence_ranked_mods["Dust law"].append(dust_arr[n_dust].name)
                    evidence_ranked_mods["E(B-V)"].append(dust_arr[n_dust].EBV)
                    z_distrib = probsarr[n_temp, n_dust, :] / jnp.trapz(probsarr[n_temp, n_dust, :], x=z_grid)
                    cum_distr = jnp.cumsum(z_distrib)
                    z_mode = z_grid[jnp.nanargmax(z_distrib)]
                    evidence_ranked_mods["zp (mode)"].append(z_mode)
                    #opa_at_z = Extinction.opacity_at_z(jnp.array([z_mode]), opa_zgrid, opacity_grid)
                    opa_at_z = jnp.array([jnp.interp(z_mode, opa_zgrid, opacity_grid[:, _col]) for _col in range(opacity_grid.shape[1])])
                    opacities = jnp.concatenate((opa_at_z, jnp.ones(len(wl_grid)-len(wls_opa))), axis=None)
                    templ_fab = Template.make_scaled_template(baseTemp_arr[n_temp].flux, filters_arr,\
                                                              dust_arr[n_dust].transmission,\
                                                              observ.AB_fluxes, observ.AB_f_errors,\
                                                              z_mode, wl_grid,\
                                                              Cosmology.distMod(cosmo, z_mode),\
                                                              opacities
                                                             )
                    templ_mab = -2.5*jnp.log10(templ_fab)-48.6
                    z_avg = jnp.trapz(z_distrib*z_grid, x=z_grid)
                    evidence_ranked_mods["average(z)"].append(z_avg)
                    evidence_ranked_mods["sigma(z)"].append(jnp.trapz(z_distrib*jnp.power(z_grid-z_avg, 2), x=z_grid))
                    _selmed = cum_distr > 0.5
                    try :
                        evidence_ranked_mods["median(z)"].append(z_grid[_selmed][0])
                    except IndexError:
                        evidence_ranked_mods["median(z)"].append(None)
                    evidence_ranked_mods["Odd ratio"].append(evs_nosplit[sorted_evs[rank]]/evs_nosplit[sorted_evs[0]])
                    evidence_ranked_mods["Bias"].append(jnp.abs(z_mode - observ.z_spec)/(1+observ.z_spec))
                    for num_f, f in enumerate(named_filts):
                        evidence_ranked_mods[f"M({f.name})"].append(templ_mab[num_f])

            if jnp.isfinite(observ.z_spec):
                p_zfix_nosplit, _n = Analysis.probs_at_fixed_z(probsarr, observ.z_spec, len(baseTemp_arr), len(extlaws_dict),\
                                                               ebv_vals, z_grid, renormalize=True, prenormalize=False)
                sorted_pzfix_flat = jnp.argsort(p_zfix_nosplit, axis=None)
                sorted_pzfix = [ jnp.unravel_index(idx, p_zfix_nosplit.shape) for idx in sorted_pzfix_flat ]
                sorted_pzfix.reverse()
                n_temp, n_dust = sorted_pzfix[0]

                df_gal.loc[i, "Best SED at z_spec"] = baseTemp_arr[n_temp].name
                df_gal.loc[i, "Best dust law at z_spec"] = dust_arr[n_dust].name
                df_gal.loc[i, "E(B-V) at z_spec"] = dust_arr[n_dust].EBV
                df_gal.loc[i, "Odd ratio"] = float(evs_nosplit[n_temp, n_dust] / evs_nosplit[mod_num, ext_num])

                if inputs['z_spec analysis']:
                    # Include more z_spec-derived properties
                    #opa_at_z = Extinction.opacity_at_z(jnp.array([observ.z_spec]), opa_zgrid, opacity_grid)
                    opa_at_z = jnp.array([jnp.interp(observ.z_spec, opa_zgrid, opacity_grid[:, _col]) for _col in range(opacity_grid.shape[1])])
                    opacities = jnp.concatenate((opa_at_z, jnp.ones(len(wl_grid)-len(wls_opa))), axis=None)

                    for rank, (n_temp, n_dust) in enumerate(sorted_pzfix[:NMOD]):
                        mods_at_z_spec["Template SED"].append(baseTemp_arr[n_temp].name)
                        mods_at_z_spec["Dust law"].append(dust_arr[n_dust].name)
                        mods_at_z_spec["E(B-V)"].append(dust_arr[n_dust].EBV)
                        z_distrib = probsarr[n_temp, n_dust, :] / jnp.trapz(probsarr[n_temp, n_dust, :], x=z_grid)
                        cum_distr = jnp.cumsum(z_distrib)
                        z_mode = z_grid[jnp.nanargmax(z_distrib)]
                        mods_at_z_spec["zp (mode)"].append(z_mode)
                        templ_fab = Template.make_scaled_template(baseTemp_arr[n_temp].flux, filters_arr,\
                                                                  dust_arr[n_dust].transmission,\
                                                                  observ.AB_fluxes, observ.AB_f_errors,\
                                                                  observ.z_spec, wl_grid,\
                                                                  Cosmology.distMod(cosmo, observ.z_spec),\
                                                                  opacities
                                                                 )
                        templ_mab = -2.5*jnp.log10(templ_fab)-48.6
                        z_avg = jnp.trapz(z_distrib*z_grid, x=z_grid)
                        mods_at_z_spec["average(z)"].append(z_avg)
                        mods_at_z_spec["sigma(z)"].append(jnp.trapz(z_distrib*jnp.power(z_grid-z_avg, 2), x=z_grid))
                        _selmed = cum_distr > 0.5
                        try :
                            mods_at_z_spec["median(z)"].append(z_grid[_selmed][0])
                        except IndexError:
                            mods_at_z_spec["median(z)"].append(None)
                        mods_at_z_spec["Odd ratio"].append(p_zfix_nosplit[sorted_pzfix[rank]]/p_zfix_nosplit[sorted_pzfix[0]])
                        mods_at_z_spec["Bias"].append(jnp.abs(z_mode - observ.z_spec)/(1+observ.z_spec))
                        for num_f, f in enumerate(named_filts):
                            mods_at_z_spec[f"M({f.name})"].append(templ_mab[num_f])
            else:
                df_gal.loc[i, "Best SED at z_spec"] = None
                df_gal.loc[i, "Best dust law at z_spec"] = None
                df_gal.loc[i, "E(B-V) at z_spec"] = None
                df_gal.loc[i, "Odd ratio"] = None

                if inputs['z_spec analysis']:
                    for rep in range(NMOD):
                        mods_at_z_spec["Template SED"].append(None)
                        mods_at_z_spec["Dust law"].append(None)
                        mods_at_z_spec["E(B-V)"].append(None)
                        mods_at_z_spec["zp (mode)"].append(None)
                        mods_at_z_spec["average(z)"].append(None)
                        mods_at_z_spec["sigma(z)"].append(None)
                        mods_at_z_spec["median(z)"].append(None)
                        mods_at_z_spec["Odd ratio"].append(None)
                        mods_at_z_spec["Bias"].append(None)
                        for num_f, f in enumerate(named_filts):
                            mods_at_z_spec[f"M({f.name})"].append(None)

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

            # distributions
            #z_phot_loc = jnp.nanargmin(chi2_arr, axis=2)
            #tempId_arr = [ temp.name for temp in baseTemp_arr ]
            #extLaw_arr = [ ext.name for ext in dust_arr ]
            #eBV_arr = [ ext.EBV for ext in dust_arr ]
            #res_dict = {'zp': z_grid, 'chi2': chi2_arr, 'mod id': tempId_arr, 'ext law': extLaw_arr, 'eBV': eBV_arr, 'min_locs': z_phot_loc}
            dict_of_results_dict[i] = {"Id": observ.num,\
                                       "Full posterior": probsarr,\
                                       f"{NMOD} most likely models": evidence_ranked_mods,\
                                       f"{NMOD} best models at z_spec": mods_at_z_spec\
                                      }
        else:
            empty_counts+=1
            empty_list.append(observ.num)
        
    debug.print("{c} empty observations : {l}", c=empty_counts, l=empty_list)

    if inputs["save results"]:
        df_gal.to_pickle(f"{inputs['run name']}_results_summary.pkl")
        with open(f"{inputs['run name']}_posteriors_dict.pkl", 'wb') as handle:
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
