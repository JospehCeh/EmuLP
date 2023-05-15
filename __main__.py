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
#import EmuLP.Cosmology as Cosmology
#import EmuLP.Filter
#import EmuLP.Galaxy
#import EmuLP.Estimator
#import EmuLP.Extinction
#import EmuLP.Template

import numpy as np
import pandas as pd
import json
import os,sys,copy
import matplotlib.pyplot as plt

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
    z_grid = np.arange(inputs['Z_GRID']['z_min'], inputs['Z_GRID']['z_max']+inputs['Z_GRID']['z_step'], inputs['Z_GRID']['z_step'])
    wl_grid = np.arange(inputs['WL_GRID']['lambda_min'], inputs['WL_GRID']['lambda_max']+inputs['WL_GRID']['lambda_step'], inputs['WL_GRID']['lambda_step'])
    extlaws_dict = inputs['Extinctions']
    extlaws_arr = np.array( [ Extinction.Extinction(key, extlaws_dict[key]) for key in extlaws_dict.keys() ] )
    
    filters_dict = inputs['Filters']
    filters_arr = np.array( [ Filter.Filter(key, filters_dict[key]["path"], filters_dict[key]["transmission"]) for key in filters_dict.keys() ] )
    N_FILT = len(filters_arr)
    
    df_filters = Filter.to_df(filters_arr, wl_grid, inputs["save results"], f"{inputs['run name']}_filters")
    templates_arr = np.array([])
    
    for temp_key in templates_dict:
        for extinction in extlaws_arr:
            for ebv in inputs['e_BV']:
                for redshift in z_grid:
                    template = Template.Template(temp_key, templates_dict[temp_key])
                    template.apply_extinc(extinction, ebv)
                    template.to_redshift(redshift, lcdm)
                    template.normalize(1000., 10000.)
                    #print(np.trapz(filters_arr[0].f_transmit(template.wavelengths), template.wavelengths))
                    template.fill_magnitudes(filters_arr)
                    #print(template.magnitudes)
                    templates_arr = np.append(templates_arr, copy.deepcopy(template))
                    del template
    
    df_templates_prop, df_templates = Template.to_df(templates_arr, filters_arr, wl_grid, inputs["save results"], f"{inputs['run name']}_templates")
                    
    estim_method = inputs['Estimator']
    if estim_method.lower() == 'chi2':
        zp_estim = Estimator.Chi2(estim_method, templates_arr)
    else:
        raise RuntimeError(f"Unimplemented estimator {estim_method}.\nPlease specify one of the following: chi2, <more to come>.")
    

    
    data_path = os.path.abspath(inputs['Dataset']['path'])
    data_ismag = (inputs['Dataset']['type'].lower() == 'm')
    
    data_file_arr = np.loadtxt(data_path)
    galaxy_arr = np.array([])
    
    print("\n")
    for i in range(data_file_arr.shape[0]):
        try:
            assert (len(data_file_arr[i,:]) == 1+2*N_FILT) or (len(data_file_arr[i,:]) == 1+2*N_FILT+1), f"At least one filter is missing in datapoint {data_file_arr[i,0]} : length is {len(data_file_arr[i,:])}, {1+2*N_FILT} values expected.\nDatapoint removed from dataset."
            if (len(data_file_arr[i,:]) == 1+2*N_FILT+1):
                gal = Galaxy.Galaxy(data_file_arr[i, 0], data_file_arr[i, 1:2*N_FILT+1], data_ismag, zs=data_file_arr[i, 2*N_FILT+1])
            else:
                gal = Galaxy.Galaxy(data_file_arr[i, 0], data_file_arr[i, 1:2*N_FILT+1], data_ismag)
            gal.estimate_zp(zp_estim, filters_arr)
            galaxy_arr = np.append(galaxy_arr, gal)
        except AssertionError:
            pass
        _progress = ((i+1)*100) // data_file_arr.shape[0]
        if _progress%2 == 1:
            #sys.stdout[-1] = (f"Estimation: {_progress}% done out of {data_file_arr.shape[0]} galaxies in dataset.")
            print(f"Estimation: {_progress}% done out of {data_file_arr.shape[0]} galaxies in dataset.", flush=True)
    df_results = Galaxy.to_df(galaxy_arr, filters_arr, inputs["save results"], f"{inputs['run name']}_results")
    f,a = plt.subplots(1,1)
    df_results.plot.scatter("True redshift", "Photometric redshift", ax=a)
    f.show()
    
    return 0

if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))
