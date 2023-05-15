#!/bin/env python3

import numpy as np

# pi
# pi = 3.14159265359 #on utilise np.pi
# c
c = 2.99792458e18 # en A/s
ckms = 2.99792458e5 # en km/s
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

class Cosmology:
    """Implementation of the functions of the cosmo class from LePhare in python"""
    def __init__(self, h0=70., om0=0.3, l0=0.7):
        self.h0 = h0
        self.om0 = om0
        self.l0 = l0
        self.omt = self.om0+self.l0
        
    def __str__(self):
        return f"LCDM cosmology with h0={self.h0} ; Om0={self.om0} ; l0={self.l0}."
        
    def __repr__(self):
        return f"<Cosmology object : h0={self.h0} ; Om0={self.om0} ; l0={self.l0}>"
    
    # Compute the distance modulus
    def distMod(self, z):
        funz = 0.
        if (z >= 1.e-10):
            funz = 5.*np.log10( self.distLum(z) ) + 25
        return funz

    # Compute the metric distance dmet in Mpc : dlum = dmet*(1+z), dang = dmet/(1+z) = dlum/(1+z)^2
    def distMet(self, z):
        dmet, ao = 0., 1.
        # case without the cosmological constant
        if (self.l0==0):
            # ao = c/(self.h0*np.sqrt(np.abs(1-self.omt)))
            # in fact we use x = ao * x(z) with x(z) from eq 8 of
            # Moscardini et al.  So we don't need to compute ao
            if (self.om0>0):
                dmet = self.om0*z-(self.om0-2)*(1-np.sqrt(1+self.om0*z))
                dmet = 2*ckms/(ao*self.h0*self.om0*self.om0*(1+z))*dmet
            else:
                dmet = ckms*z*(1.+z/2.)/(self.h0*(1+z))

        elif (self.om0<1 and self.l0 != 0):
            _sum = 0.
            dz = z/50.
            for i in range(50):
                zi = (i+0.5)*dz
                Ez = np.sqrt(self.om0*np.power((1.+zi),3.)+(1-self.om0-self.l0)*np.power((1.+zi),2.)+self.l0)
                _sum = _sum + dz/Ez
            dmet = ckms/(self.h0*ao) * _sum
        else:
            raise RuntimeError(f"Cosmology not included : h0={self.h0}, Om0={self.om0}, l0={self.l0}")
        return dmet
        
    def distLum(self, z):
        return self.distMet(z)*(1+z)
    
    def distAng(self, z):
        return self.distMet(z)/(1+z)

    ## Compute cosmological time from z=infinty  to z
    ## as a function of cosmology.  Age given in year !!
    ## Note : use lambda0 non zero if Omega_o+lambda_o=1
    def time(self, z):
        timy = 0.
        val = 0.
        hy = self.h0*1.0224e-12

        if (np.abs(self.om0-1)<1.e-6 and self.l0==0):
            timy = 2.*np.power((1+z),-1.5)/(3*hy)
        elif (self.om0==0 and self.l0==0): 
            timy = 1./(hy*(1+z))
        elif (self.om0<1 and self.om0>0 and self.l0==0):
            val = (self.om0*z-self.om0+2.) / (self.om0*(1+z))
            timy = 2.*np.sqrt((1-self.om0)*(self.om0*z+1))/(self.om0*(1+z))
            timy = timy - np.log10(val+np.sqrt(val*val-1))
            timy = timy*self.om0/(2.*hy*np.power((1-self.om0),1.5))

        elif(self.om0>1 and self.l0==0):
            timy = np.arccos((self.om0*z-self.om0+2.)/(self.om0*(1+z)))
            timy = timy-2*np.sqrt((self.om0-1)*(self.om0*z+1))/(self.om0*(1+z))
            timy = timy*self.om0/(2*hy*np.power((self.om0-1),1.5))

        elif(self.om0<1 and np.abs(self.om0+self.l0-1)<1.e-5 ):
            val = np.sqrt(1-self.om0)/(np.sqrt(self.om0)*np.power((1+z),1.5))
            timy = np.log(val+np.sqrt(val*val+1))
            timy = timy*2./(3.*hy*np.sqrt(1-self.om0))

        else:
            raise RuntimeError(f"Not the right cosmology to derive the time.")
        
        return timy

    # Two possible grid in redshift : linear or in (1+z)
    # Possible now to define a minimum redshift to be considered
    def zgrid(self, gridType, dz, zmin, zmax):
        assert zmax < zmin, "You are probably using the old parametrisation of Z_STEP since Z MIN > Z MAX in Z_STEP."
        z = np.array([])
        # first redshift at 0
        z = np.append(z, 0.)

        # Start at zmin
        if(zmin > 0.):
            z = np.append(z, zmin)
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
                    z = np.append(z, zinter)
        # Linear grid in redshift
        else:
            while (zinter<zmax):
                # Step in dz
                zinter = zmin + count * dz
                # keep only in the redshift range zmin-zmax defined in Z_STEP
                if(zinter>zmin and zinter<zmax):
                    z = np.append(z, zinter)
                count+=1
        z = np.append(z, zmax)
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
            up = np.where(gridz >= red)[0]
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
