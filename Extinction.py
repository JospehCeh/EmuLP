#!/bin/env python3

import numpy as np
import jax.numpy as jnp
from jax import vmap
#from scipy.interpolate import interp1d
import os
import munch
from functools import partial
from collections import namedtuple

DustLaw = namedtuple('DustLaw', ['name', 'EBV', 'transmission'])

class Extinction(munch.Munch):
    """Class for generating extinction law functions"""
    def __init__(self, ident, extincfile):
        super().__init__()
        self.id = ident
        #self.path = os.path.abspath(extincfile)
        self.wavelengths, self.coeff = np.loadtxt(os.path.abspath(extincfile), unpack=True)
        #self.f_coeff = lambda x : jnp.interp(x, self.wavelengths, self.coeff, left=0., right=0., period=None)
            #interp1d(self.wavelengths, self.coeff, bounds_error=False, fill_value=0.)
    '''
    def name(self, dico):
        _names = [ n for n in dico ]
        _ids = [ dico[n]['id'] for n in dico ]
        return _names[_ids == self.id]
    
    def path(self, dico):
        _paths = [ dico[n]['path'] for n in dico ]
        _ids = [ dico[n]['id'] for n in dico ]
        return _paths[_ids == self.id]
    '''
    def name(self, dico):
        return dico[self.id]['name']
    
    def path(self, dico):
        return dico[self.id]['path'] 
    
    def extinct_func(self, lambdas, ebv):
        f_coeff = lambda x : jnp.interp(x, self.wavelengths, self.coeff, left=0., right=0., period=None)
        transmits = calc_transmit( ebv, f_coeff(lambdas) )
        f_trans = lambda x : jnp.interp(x, lambdas, transmits,\
                                        left=0., right=0., period=None)
            #interp1d(lambdas, transmits, bounds_error=False, fill_value=0.)
        return f_trans
        
    def __str__(self):
        return f"Extinction law {self.id}" #from file {self.path}"
        
    def __repr__(self):
        return f"<Extinction object : id={self.id}" #; path={self.path}>"


def load_extinc(ident, extincfile, ebv, wl_grid):
    wls, co = np.loadtxt(os.path.abspath(extincfile), unpack=True)
    wavelengths, coeff = jnp.array(wls), jnp.array(co)
    _trans = calc_transmit(ebv, coeff)
    transmits = jnp.interp(wl_grid, wavelengths, _trans, left=0., right=0., period=None)
    return transmits
    
@partial(vmap, in_axes=(None, 0))
def calc_transmit(ebv,c):
    return jnp.power(10., -0.4*ebv*c)