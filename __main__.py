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
    
    ebvs_in_use = jnp.array([d.EBV for d in dust_arr])
    laws_in_use = jnp.array([0 if d.name == "Calzetti" else 1 for d in dust_arr])
    
    _old_dir = os.getcwd()
    _path = os.path.abspath(__file__)
    _dname = os.path.dirname(_path)
    os.chdir(_dname)
    opa_path = os.path.abspath(inputs['Opacity'])
    #ebv_prior_file = inputs['E(B-V) prior file']
    #ebv_prior_df = pd.read_pickle(ebv_prior_file)
    #cols_to_stack = tuple(ebv_prior_df[col].values for col in ebv_prior_df.columns)
    #ebv_prior_arr = jnp.column_stack(cols_to_stack)
    os.chdir(_old_dir)
    
    _selOpa = (wl_grid < 1300.)
    wls_opa = wl_grid[_selOpa]
    opa_zgrid, opacity_grid = Extinction.load_opacity(opa_path, wls_opa)
    extrap_ones = jnp.ones((len(z_grid), len(wl_grid)-len(wls_opa)))
    
    print('Photometric redshift estimation :')
    df_gal = pd.DataFrame()
    dict_of_results_dict = {}
    
    #@partial(jit, static_argnums=(1,2))
    def estim_zp(observ, prior=True, ebv_prior=False, jaxcosmo=False, id_prior_band=4):
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
            if ebv_prior :
                if jaxcosmo:
                    chi2_arr = Galaxy.est_chi2_prior_ebv_jaxcosmo(f,\
                                                                  f_err,\
                                                                  z_grid, baseFluxes_arr, extlaws_arr, laws_in_use, ebvs_in_use,\
                                                                  filts, cosmo, wl_grid,\
                                                                  interpolated_opacities, prior_band)
                else:
                    chi2_arr = Galaxy.est_chi2_prior_ebv(f,\
                                                         f_err,\
                                                         z_grid, baseFluxes_arr, extlaws_arr, laws_in_use, ebvs_in_use,\
                                                         filts, cosmo, wl_grid,\
                                                         interpolated_opacities, prior_band)
            else:
                if jaxcosmo:
                    chi2_arr = Galaxy.est_chi2_prior_jaxcosmo(f,\
                                                              f_err,\
                                                              z_grid, baseFluxes_arr, extlaws_arr, ebvs_in_use,\
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
            chi2_arr, z_phot_loc = estim_zp(observ, inputs['prior'], inputs['E(B-V) prior'], inputs['Cosmology']['jax-cosmo'])
            
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

            if len(extlaws_dict)==1 and len(ebv_vals)==1:
                probsarr, norm = Analysis.probability_distrib_noDust(chi2_arr, len(baseTemp_arr), z_grid)
            elif len(extlaws_dict)==1:
                probsarr, norm = Analysis.probability_distrib_oneLaw(chi2_arr, len(baseTemp_arr), ebv_vals, z_grid)
            elif len(ebv_vals)==1:
                probsarr, norm = Analysis.probability_distrib_oneEBV(chi2_arr, len(baseTemp_arr), len(extlaws_dict), z_grid)
            else:
                probsarr, norm = Analysis.probability_distrib(chi2_arr, len(baseTemp_arr), len(extlaws_dict), ebv_vals, z_grid)
            
            while abs(1-norm)>1.0e-3 :
                chi2_arr = chi2_arr + 2*jnp.log(norm)
                if len(extlaws_dict)==1 and len(ebv_vals)==1:
                    probsarr, norm = Analysis.probability_distrib_noDust(chi2_arr, len(baseTemp_arr), z_grid)
                elif len(extlaws_dict)==1:
                    probsarr, norm = Analysis.probability_distrib_oneLaw(chi2_arr, len(baseTemp_arr), ebv_vals, z_grid)
                elif len(ebv_vals)==1:
                    probsarr, norm = Analysis.probability_distrib_oneEBV(chi2_arr, len(baseTemp_arr), len(extlaws_dict), z_grid)
                else:
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
                    lst = [0.]+[jnp.trapz(z_distrib[:j+1], x=z_grid[:j+1]) for j in range(len(z_distrib)-1)]
                    cum_distr = jnp.array(lst) #jnp.cumsum(z_distrib)
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
                    evidence_ranked_mods["Odd ratio"].append(evs_nosplit[n_temp, n_dust]/evs_nosplit[mod_num, ext_num])
                    evidence_ranked_mods["Bias"].append(z_mode - observ.z_spec)
                    #evidence_ranked_mods["Bias"].append(jnp.abs(z_mode - observ.z_spec)/(1+observ.z_spec))
                    for num_f, f in enumerate(named_filts):
                        evidence_ranked_mods[f"M({f.name})"].append(templ_mab[num_f])

            if jnp.isfinite(observ.z_spec):
                if len(extlaws_dict)==1 and len(ebv_vals)==1:
                    p_zfix_nosplit, _n = Analysis.probs_at_fixed_z_noDust(probsarr, observ.z_spec, len(baseTemp_arr), z_grid,\
                                                                          renormalize=True, prenormalize=False)
                elif len(extlaws_dict)==1:
                    p_zfix_nosplit, _n = Analysis.probs_at_fixed_z_oneLaw(probsarr, observ.z_spec, len(baseTemp_arr),\
                                                                          ebv_vals, z_grid, renormalize=True, prenormalize=False)
                elif len(ebv_vals)==1:
                    p_zfix_nosplit, _n = Analysis.probs_at_fixed_z_oneEBV(probsarr, observ.z_spec, len(baseTemp_arr), len(extlaws_dict),\
                                                                          z_grid, renormalize=True, prenormalize=False)
                else:
                    p_zfix_nosplit, _n = Analysis.probs_at_fixed_z(probsarr, observ.z_spec, len(baseTemp_arr), len(extlaws_dict),\
                                                                   ebv_vals, z_grid, renormalize=True, prenormalize=False)
                sorted_pzfix_flat = jnp.argsort(p_zfix_nosplit, axis=None)
                sorted_pzfix = [ jnp.unravel_index(idx, p_zfix_nosplit.shape) for idx in sorted_pzfix_flat ]
                sorted_pzfix.reverse()
                n_temp, n_dust = sorted_pzfix[0]

                df_gal.loc[i, "Best SED at z_spec"] = baseTemp_arr[n_temp].name
                df_gal.loc[i, "Best dust law at z_spec"] = dust_arr[n_dust].name
                df_gal.loc[i, "E(B-V) at z_spec"] = dust_arr[n_dust].EBV
                _z_dist_zs = probsarr[n_temp, n_dust, :] / jnp.trapz(probsarr[n_temp, n_dust, :], x=z_grid)
                _mode_zs = z_grid[jnp.nanargmax(_z_dist_zs)]
                _mean_zs = jnp.trapz(_z_dist_zs*z_grid, x=z_grid)
                df_gal.loc[i, "Mode of best model at z_spec"] = _mode_zs
                df_gal.loc[i, "Mean of best model at z_spec"] = _mean_zs
                df_gal.loc[i, "Odd ratio of best model at z_spec"] = float(evs_nosplit[n_temp, n_dust] / evs_nosplit[mod_num, ext_num])

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
                        lst = [0.]+[jnp.trapz(z_distrib[:j+1], x=z_grid[:j+1]) for j in range(len(z_distrib)-1)]
                        cum_distr = jnp.array(lst) #jnp.cumsum(z_distrib)
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
                        mods_at_z_spec["Odd ratio"].append(evs_nosplit[n_temp, n_dust]/evs_nosplit[mod_num, ext_num])
                        mods_at_z_spec["Bias"].append(z_mode - observ.z_spec)
                        #mods_at_z_spec["Bias"].append(jnp.abs(z_mode - observ.z_spec)/(1+observ.z_spec))
                        for num_f, f in enumerate(named_filts):
                            mods_at_z_spec[f"M({f.name})"].append(templ_mab[num_f])
            else:
                df_gal.loc[i, "Best SED at z_spec"] = None
                df_gal.loc[i, "Best dust law at z_spec"] = None
                df_gal.loc[i, "E(B-V) at z_spec"] = None
                df_gal.loc[i, "Mode of best model at z_spec"] = None
                df_gal.loc[i, "Mean of best model at z_spec"] = None
                df_gal.loc[i, "Odd ratio of best model at z_spec"] = None

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

if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))
