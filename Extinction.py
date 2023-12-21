#!/bin/env python3

import numpy as np
import jax.numpy as jnp
from jax import vmap, jit
#from scipy.interpolate import interp1d
import os
from functools import partial
from collections import namedtuple

DustLaw = namedtuple('DustLaw', ['name', 'EBV', 'transmission'])

def load_extinc(extincfile, ebv, wl_grid):
    wls, co = np.loadtxt(os.path.abspath(extincfile), unpack=True)
    wavelengths, coeff = jnp.array(wls), jnp.array(co)
    _trans = calc_transmit(ebv, coeff)
    transmits = jnp.interp(wl_grid, wavelengths, _trans, left=_trans[0], right=1., period=None)
    return transmits

def load_opacity(opa_list_file, lambdas):
    z_opas, files_opa = np.loadtxt(os.path.abspath(opa_list_file), dtype=str, unpack=True)
    z_opas = jnp.array([float(z_opa) for z_opa in z_opas])
    opa_arr = np.empty((len(z_opas), len(lambdas)))
    for _fid, _file in enumerate(files_opa):
        opa_ascii_file = os.path.join(os.path.dirname(opa_list_file), _file)
        wls, op = np.loadtxt(opa_ascii_file, unpack=True)
        op_interp = np.interp(lambdas, wls, op, left=1., right=1., period=None)
        opa_arr[_fid, :] = op_interp
    return z_opas, jnp.array(opa_arr)

@jit
@partial(vmap, in_axes=(0, None, None))
def opacity_at_z(z, z_grid_opa, opacity_arr):
    #opa_at_z = jnp.empty(opacity_arr.shape[1])
    #for _col in jnp.arange(opacity_arr.shape[1]):
    #    opa_at_z = opa_at_z.at(_col).set(jnp.interp(z, z_grid_opa, opacity_arr[:, _col]))
    opa_at_z = jnp.array([jnp.interp(z, z_grid_opa, opacity_arr[:, _col]) for _col in range(opacity_arr.shape[1])])
    return opa_at_z
    
@partial(vmap, in_axes=(None, 0))
def calc_transmit(ebv,c):
    return jnp.power(10., -0.4*ebv*c)