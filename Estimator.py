#!/bin/env python3

import numpy as np
from scipy.interpolate import interp1d
import pandas as pd

from EmuLP import Cosmology
from EmuLP import Galaxy
from EmuLP import Filter

class Estimator:
    def __init__(self, method):
        self.method = method
        
    def __str__(self):
        return f"Photo-z {self.type} estimator using {self.method} method."
        
    def __repr__(self):
        return f"<Estimator object : type={self.type}, method={self.method}>"
        
    def estimate_zp(self, galaxy):
        raise NotImplementedError("Please specify which estimator you wan to use: chi2, <more to come>")
        
class Chi2(Estimator):
    def __init__(self, method, templates):
        super().__init__(method)
        self.type = 'Template Fitting'
        self.templates = templates
    
    def estimate_zp(self, galaxy, filters):
        _chi2_init, _scale = galaxy.chi2(filters, self.templates[0])
        _chi2_arr = np.array([_chi2_init])
        _chi2_arr = np.append(_chi2_arr, [ _chi for _chi, _sc in [ galaxy.chi2(filters, _templ, _scale, rescale_all=False) for _templ in self.templates[1:] ] ])
        _sel = np.isfinite(_chi2_arr)
        _minChi2 = np.nanargmin(_chi2_arr[_sel])
        #print(f"Min chi2 : {_minChi2} for template {np.where(_chi2_arr == _minChi2)[0]}")
        _templatesMin = (self.templates[_sel])[_minChi2]
        
        zp = _templatesMin.redshift
        tempId = _templatesMin.name
        extLaw = _templatesMin.extinc_law
        eBV = _templatesMin.ebv
        chi2 = (_chi2_arr[_sel])[_minChi2]
        
        #zp = [_temp.redshift for _temp in _templatesMin ][0]
        #tempId = [_temp.name for _temp in _templatesMin ][0]
        #extLaw = [_temp.extinc_law for _temp in _templatesMin ][0]
        #eBV = [_temp.ebv for _temp in _templatesMin ][0]
        
        return zp, tempId, extLaw, eBV, chi2
        
        
