#!/bin/env python3

from functools import partial
import jax.numpy as jnp
from jax import jit, vmap
#from scipy.interpolate import interp1d
import pandas as pd

from EmuLP import Cosmology
from EmuLP import Galaxy
from EmuLP import Filter
#import munch

'''
class Estimator(): #munch.Munch):
    def __init__(self, method):
        #super().__init__()
        self.method = method
        
    def __str__(self):
        return f"Photo-z {self.type} estimator using {self.method} method."
        
    def __repr__(self):
        return f"<Estimator object : type={self.type}, method={self.method}>"
    
    def __eq__(self, other):
        return (self.method.lower() == other.method.lower())
    
    def __ne__(self, other):
        return not self.__eq__(other)
    
    def __hash__(self):
        return hash(self.method)
        
    def estimate_zp(self, galaxy):
        raise NotImplementedError("Please specify which estimator you wan to use: chi2, <more to come>")
'''
        
class Chi2():
    def __init__(self, templates):
        #super().__init__(method)
        #self.method = method
        #self.type = 'Template Fitting'
        self.templates = tuple(templates)
        
    def __str__(self):
        return f"Photo-z Template Fitting estimator using chi2 method."
        
    def __repr__(self):
        return f"<Estimator object : type=Template Fitting, method=chi2>"
    
    def __eq__(self, other):
        #return (self.method.lower() == other.method.lower()) and (self.templates == other.templates)
        #return (self.templates == other.templates)
        return False
    
    def __ne__(self, other):
        return not self.__eq__(other)
    
    def __hash__(self):
        #hashval = hash((7, *self.templates))
        #return hashval
        return 0
        
    #@partial(vmap, in_axes=(None, 0, None))
    def estimate_zp(self, galaxy, filters):
        _chi2_arr = galaxy.chi2(filters, self.templates) #jnp.array([ _chi for _chi, _sc in [ galaxy.chi2(filters, _templ) for _templ in self.templates ] ])
        _sel = jnp.isfinite(_chi2_arr)
        _minChi2 = jnp.nanargmin(_chi2_arr[_sel])
        #print(f"Min chi2 : {_minChi2} for template {np.where(_chi2_arr == _minChi2)[0]}")
        _templatesMin = (self.templates[_sel])[_minChi2]
        
        # point estimate
        _zp = _templatesMin.redshift
        tempId = _templatesMin.id
        extLaw = _templatesMin.extinc_law
        eBV = _templatesMin.ebv
        _chi2 = (_chi2_arr[_sel])[_minChi2]
        
        # distributions
        chi2_arr = _chi2_arr[_sel]
        templates_arr = self.templates[_sel]
        zp_arr = [_temp.redshift for _temp in templates_arr]
        tempId_arr = [_temp.id for _temp in templates_arr ]
        extLaw_arr = [_temp.extinc_law for _temp in templates_arr ]
        eBV_arr = [_temp.ebv for _temp in templates_arr ]
        
        res_dict = {'zp': zp_arr, 'chi2': chi2_arr, 'mod id': tempId_arr, 'ext law': extLaw_arr, 'eBV': eBV_arr}
        
        '''
        galdic = galaxy.toDict()
        galdic["zp"] = zp
        galdic["template"] = tempId
        galdic["extinction_law"] = extLaw
        galdic["ebv"] = eBV
        galdic["chi2"] = chi2
        galdic["results_dict"] = res_dict
        '''
        
        resgal = Galaxy.return_updated_galaxy(galaxy, zp=_zp, chi2=_chi2, template=tempId,\
                                              extinction_law=extLaw, ebv=eBV, results_dict=res_dict)
        
        return resgal
        
        
