#!/bin/env python3

from functools import partial
import jax_cosmo as jc
import jax.numpy as jnp
from jax import vmap, jit
import munch
from collections import namedtuple

# pi
# pi = 3.14159265359 #on utilise np.pi
# c
ckms = jc.constants.c # en km/s
c = ckms * 1.0e13 #2.99792458e18 # en A/s
# h
hplanck = 6.62606957e-34
# k Boltzmann
kboltzmann = 1.3806488e-23
# L solar
Lsol = 3.826e33
# pc en cm
pc = 3.086e18
# hc from Cedric
hc = 12398.42 # [eV.A]
# f_ga from Cedric
f_ga = 1

Cosmo = namedtuple('Cosmo', ['h0', 'om0', 'l0', 'omt'])
PriorParams = namedtuple('PriorParams', ['mod', 'zot', 'kt', 'alpt0', 'pcal', 'ktf', 'ft', 'nuv_range'])

# P(T|m0)
ktf = jnp.array([0.47165, 0.30663, 0.12715, -0.34437])
ft = jnp.array([0.43199, 0.07995, 0.31162, 0.21220])

prior_params_set = (PriorParams(0, 0.45181, 0.13677, 3.33078, 0.89744, ktf[0], ft[0], (4.25, jnp.inf)),\
                    PriorParams(1, 0.16560, 0.12983, 1.42815, 0.90868, ktf[1], ft[1], (3.19, 4.25)),\
                    PriorParams(2, 0.21072, 0.14008, 1.58310, 0.89747, ktf[2], ft[2], (1.9, 3.19)),\
                    PriorParams(3, 0.20418, 0.13773, 1.34500, 0.91760, ktf[3], ft[3], (-jnp.inf, 1.9)),\
                   )

prior_pars_E_S0, prior_pars_Sbc, prior_pars_Scd, prior_pars_Irr = prior_params_set

@jit
def prior_zot(nuvk):
    val = prior_pars_Irr.zot +\
            (prior_pars_Scd.zot - prior_pars_Irr.zot)*jnp.heaviside(nuvk-prior_pars_Scd.nuv_range[0], 0)+\
            (prior_pars_Sbc.zot - prior_pars_Scd.zot)*jnp.heaviside(nuvk-prior_pars_Sbc.nuv_range[0], 0)+\
            (prior_pars_E_S0.zot - prior_pars_Sbc.zot)*jnp.heaviside(nuvk-prior_pars_E_S0.nuv_range[0], 0)
    return val

@jit
def prior_alpt0(nuvk):
    val = prior_pars_Irr.alpt0 +\
            (prior_pars_Scd.alpt0 - prior_pars_Irr.alpt0)*jnp.heaviside(nuvk-prior_pars_Scd.nuv_range[0], 0)+\
            (prior_pars_Sbc.alpt0 - prior_pars_Scd.alpt0)*jnp.heaviside(nuvk-prior_pars_Sbc.nuv_range[0], 0)+\
            (prior_pars_E_S0.alpt0 - prior_pars_Sbc.alpt0)*jnp.heaviside(nuvk-prior_pars_E_S0.nuv_range[0], 0)
    return val

@jit
def prior_kt(nuvk):
    val = prior_pars_Irr.kt +\
            (prior_pars_Scd.kt - prior_pars_Irr.kt)*jnp.heaviside(nuvk-prior_pars_Scd.nuv_range[0], 0)+\
            (prior_pars_Sbc.kt - prior_pars_Scd.kt)*jnp.heaviside(nuvk-prior_pars_Sbc.nuv_range[0], 0)+\
            (prior_pars_E_S0.kt - prior_pars_Sbc.kt)*jnp.heaviside(nuvk-prior_pars_E_S0.nuv_range[0], 0)
    return val

@jit
def prior_pcal(nuvk):
    val = prior_pars_Irr.pcal +\
            (prior_pars_Scd.pcal - prior_pars_Irr.pcal)*jnp.heaviside(nuvk-prior_pars_Scd.nuv_range[0], 0)+\
            (prior_pars_Sbc.pcal - prior_pars_Scd.pcal)*jnp.heaviside(nuvk-prior_pars_Sbc.nuv_range[0], 0)+\
            (prior_pars_E_S0.pcal - prior_pars_Sbc.pcal)*jnp.heaviside(nuvk-prior_pars_E_S0.nuv_range[0], 0)
    return val

@jit
def prior_ktf(nuvk):
    val = prior_pars_Irr.mod +\
            (prior_pars_Scd.ktf - prior_pars_Irr.ktf)*jnp.heaviside(nuvk-prior_pars_Scd.nuv_range[0], 0)+\
            (prior_pars_Sbc.ktf - prior_pars_Scd.ktf)*jnp.heaviside(nuvk-prior_pars_Sbc.nuv_range[0], 0)+\
            (prior_pars_E_S0.ktf - prior_pars_Sbc.ktf)*jnp.heaviside(nuvk-prior_pars_E_S0.nuv_range[0], 0)
    return val

@jit
def prior_ft(nuvk):
    val = prior_pars_Irr.mod +\
            (prior_pars_Scd.ft - prior_pars_Irr.ft)*jnp.heaviside(nuvk-prior_pars_Scd.nuv_range[0], 0)+\
            (prior_pars_Sbc.ft - prior_pars_Scd.ft)*jnp.heaviside(nuvk-prior_pars_Sbc.nuv_range[0], 0)+\
            (prior_pars_E_S0.ft - prior_pars_Sbc.ft)*jnp.heaviside(nuvk-prior_pars_E_S0.nuv_range[0], 0)
    return val

class Cosmology(munch.Munch):
    """Implementation of the functions of the cosmo class from LePhare in python"""
    def __init__(self, h0=70., om0=0.3, l0=0.7, cosmodict=None):
        if cosmodict is None:
            super().__init__()
            self.h0 = h0
            self.om0 = om0
            self.l0 = l0
            self.omt = self.om0+self.l0
            self.name = f"LCDM_h0={self.h0}-Om0={self.om0}-l0={self.l0}"
        else:
            super().__init__(cosmodict)
        
    def __str__(self):
        return f"Cosmology object with h0={self.h0}"
        
    def __repr__(self):
        return f"<Cosmology object : h0={self.h0}>"

# Compute the metric distance dmet in Mpc : dlum = dmet*(1+z), dang = dmet/(1+z) = dlum/(1+z)^2
@partial(jit, static_argnums=0)
def distMet(cosmo, z):
    dmet, ao = 0., 1.
    # case without the cosmological constant
    if (cosmo.l0==0):
        # ao = c/(self.h0*np.sqrt(np.abs(1-self.omt)))
        # in fact we use x = ao * x(z) with x(z) from eq 8 of
        # Moscardini et al.  So we don't need to compute ao
        if (cosmo.om0>0):
            dmet = cosmo.om0*z-(cosmo.om0-2)*(1-jnp.sqrt(1+cosmo.om0*z))
            dmet = 2*ckms/(ao*cosmo.h0*cosmo.om0*cosmo.om0*(1+z))*dmet
        else:
            dmet = ckms*z*(1.+z/2.)/(cosmo.h0*(1+z))

    elif (cosmo.om0<1 and cosmo.l0 != 0):
        #_sum = 0.
        dz = z/50.
        zi = jnp.linspace(0.5*dz, z, num=50)
        Ez = jnp.power((cosmo.om0*jnp.power((1.+zi),3.)+(1-cosmo.om0-cosmo.l0)*jnp.power((1.+zi),2.)+cosmo.l0), -0.5)
        _sum = jnp.trapz(Ez, zi)
        #for i in range(50):
        #    zi = (i+0.5)*dz
        #    Ez = jnp.sqrt(cosmo.om0*jnp.power((1.+zi),3.)+(1-cosmo.om0-cosmo.l0)*jnp.power((1.+zi),2.)+cosmo.l0)
        #    _sum = _sum + dz/Ez
        dmet = ckms/(cosmo.h0*ao) * _sum
    else:
        raise RuntimeError(f"Cosmology not included : h0={cosmo.h0}, Om0={cosmo.om0}, l0={cosmo.l0}")
    return dmet

@partial(jit, static_argnums=0)
def distLum(cosmo, z):
    return distMet(cosmo, z) * (1+z)

@partial(jit, static_argnums=0)
def distAng(cosmo, z):
    return self.distMet(cosmo, z) / (1+z)
    
# Compute the distance modulus
@partial(jit, static_argnums=0)
def distMod(cosmo, z):
    #funz = 0.
    #if (z >= 1.e-10):
    funz = 5.*jnp.log10( distLum(cosmo, z)*1.0e6 ) - 5
    return funz

@partial(jit, static_argnums=0)
def nz_prior_params(nuvk):
    if (nuvk>4.25):
        # Case E/S0
        # Color UV-K of PHOTO_230506/El_cww.sed.resample.new.resample15.inter 
        mod = 0
        zot = 0.45181
        kt = 0.13677
        alpt0 = 3.33078
        pcal = 0.89744
    elif (nuvk>3.19):
        # Case Sbc
        # Color UV-K of PHOTO_230506/Sbc_cww.sed.resample.new.resample8.inter 
        mod = 1
        zot = 0.16560
        kt = 0.12983 
        alpt0 = 1.42815
        pcal = 0.90868
    elif (nuvk>1.9):
        # Case Scd
        # Color UV-K of PHOTO_230506/Scd_cww.sed.resample.new.resample7.inter  -19.4878 + 21.1501
        mod = 2
        zot = 0.21072
        kt = 0.14008
        alpt0 = 1.58310
        pcal = 0.89747
    else:
        # Case Irr
        mod = 3
        zot = 0.20418
        kt = 0.13773 
        alpt0 = 1.34500
        pcal = 0.91760
    return alpt0, zot, kt, pcal

@jit
def nz_prior_core(z, imag, alpt0, zot, kt, pcal, ktf_m, ft_m):
    kk = imag-20.
    zmax = zot + kt*kk
    pz = jnp.power(z,alpt0)*jnp.exp(-jnp.power((z/zmax),alpt0))

    # Ratio for each type
    rappSum = jnp.sum(ft*jnp.exp(-ktf*kk))
    rapp = ft_m*jnp.exp(-ktf_m*kk)

    # Normalisation of the probability function
    pcal=jnp.power(zmax,alpt0+1)/alpt0*pcal

    # Final value
    val = pz/pcal * rapp/rappSum
    return val

## Compute cosmological time from z=infinty  to z
## as a function of cosmology.  Age given in year !!
## Note : use lambda0 non zero if Omega_o+lambda_o=1
def time(cosmo, z):
    timy = 0.
    val = 0.
    hy = cosmo.h0*1.0224e-12

    if (jnp.abs(cosmo.om0-1)<1.e-6 and cosmo.l0==0):
        timy = 2.*jnp.power((1+z),-1.5)/(3*hy)
    elif (cosmo.om0==0 and cosmo.l0==0): 
        timy = 1./(hy*(1+z))
    elif (cosmo.om0<1 and cosmo.om0>0 and cosmo.l0==0):
        val = (cosmo.om0*z-cosmo.om0+2.) / (cosmo.om0*(1+z))
        timy = 2.*jnp.sqrt((1-cosmo.om0)*(cosmo.om0*z+1))/(cosmo.om0*(1+z))
        timy = timy - jnp.log10(val+jnp.sqrt(val*val-1))
        timy = timy*cosmo.om0/(2.*hy*np.power((1-cosmo.om0),1.5))

    elif(cosmo.om0>1 and cosmo.l0==0):
        timy = jnp.arccos((cosmo.om0*z-cosmo.om0+2.)/(cosmo.om0*(1+z)))
        timy = timy-2*jnp.sqrt((cosmo.om0-1)*(cosmo.om0*z+1))/(cosmo.om0*(1+z))
        timy = timy*cosmo.om0/(2*hy*jnp.power((cosmo.om0-1),1.5))

    elif(cosmo.om0<1 and jnp.abs(cosmo.om0+cosmo.l0-1)<1.e-5 ):
        val = jnp.sqrt(1-cosmo.om0)/(jnp.sqrt(cosmo.om0)*jnp.power((1+z),1.5))
        timy = jnp.log(val+jnp.sqrt(val*val+1))
        timy = timy*2./(3.*hy*jnp.sqrt(1-cosmo.om0))

    else:
        raise RuntimeError(f"Not the right cosmology to derive the time.")
    return timy
    
def make_jcosmo(H0):
    return jc.Planck15(h=H0/100.)

# JAX versions
@partial(jit, static_argnums=0)
def calc_distM(cosm, z):
    _a = jc.utils.z2a(z)
    d_M = cosm.h * jc.background.transverse_comoving_distance(cosm, _a)
    return d_M

@partial(jit, static_argnums=0)
def calc_distLum(cosm, z):
    return (1.+z) * calc_distM(cosm, z)

@partial(jit, static_argnums=0)
def calc_distAng(cosm, z):
    _a = jc.utils.z2a(z)
    d_Ang = cosm.h * jc.background.angular_diameter_distance(cosm, _a)
    return d_Ang

# Compute the distance modulus
@partial(jit, static_argnums=0)
def calc_distMod(cosm, z):
    funz = 5.*jnp.log10( calc_distLum(cosm, z)*1.0e6 ) - 5
    return funz

# Unused functions translated from LEPHARE.
'''
    # Two possible grid in redshift : linear or in (1+z)
    # Possible now to define a minimum redshift to be considered
    def zgrid(self, gridType, dz, zmin, zmax):
        assert zmax < zmin, "You are probably using the old parametrisation of Z_STEP since Z MIN > Z MAX in Z_STEP."
        z = jnp.array([])
        # first redshift at 0
        z = jnp.append(z, 0.)

        # Start at zmin
        if(zmin > 0.):
            z = jnp.append(z, zmin)
        count = 1
        zinter = zmin
        # Define a vector with the redshift grid according to the given type
        # grid in dz*(1+z)
        if gridType == 1 :
            while (zinter<zmax):
                # Step in dz*(1+z)
                zinter = zinter+(1.+zinter)*dz
                # keep only in the redshift range zmin-zmax defined in Z_STEP
                if (zinter>zmin and zinter<zmax):
                    z = jnp.append(z, zinter)
        # Linear grid in redshift
        else:
            while (zinter<zmax):
                # Step in dz
                zinter = zmin + count * dz
                # keep only in the redshift range zmin-zmax defined in Z_STEP
                if(zinter>zmin and zinter<zmax):
                    z = jnp.append(z, zinter)
                count+=1
        z = jnp.append(z, zmax)
        return z

    def indexz(self, red, gridz):
        #gridz is assumed to be sorted
        #case 1 : red <= gridz[0] : return index 0
        idz = 0
        if(red <= gridz[0]):
            idz = 0
        #case 2 : red >= gridz[size-1] : return index size-1
        elif(red >= gridz[-1]):
            idz = gridz.size-1
        else :
            up = jnp.where(gridz >= red)[0]
            #case 3 : red = gridz[k] : return k
            if (gridz[up] == red):
                idz = up
            # case 4 : gridz[0] < red < gridz[size-1] : find closest match in gridz and return its index 
            else:
                low = up-1
                if( abs(up - red) <= abs(red - low) ):
                    idz = up
                else:
                    idz = low
        return int(idz)
'''