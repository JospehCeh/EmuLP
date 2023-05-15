#!/bin/env python3
import numpy as np
from scipy.interpolate import interp1d
import os

class Extinction:
    """Class for generating extinction law functions"""
    def __init__(self, name, extincfile):
        self.name = name
        self.path = os.path.abspath(extincfile)
        self.wavelengths, self.coeff = np.loadtxt(self.path, unpack=True)
        self.f_coeff = interp1d(self.wavelengths, self.coeff, bounds_error=False, fill_value=0.)
    
    def extinct_func(self, lambdas, ebv):
        transmits = np.array([ np.power(10., -0.4*ebv*c) for c in self.f_coeff(lambdas) ])
        f_trans = interp1d(lambdas, transmits, bounds_error=False, fill_value=0.)
        return f_trans
        
    def __str__(self):
        return f"Extinction law {self.name} from file {self.path}"
        
    def __repr__(self):
        return f"<Extinction object : name={self.name} ; path={self.path}>"

