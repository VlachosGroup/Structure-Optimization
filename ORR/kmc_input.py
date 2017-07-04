# -*- coding: utf-8 -*-
"""
Created on Mon Jul 03 14:34:42 2017

@author: mpnun
"""


OHslope = 0.1915   	  
OHint = -3.8166     	  
OOHslope = 0.1697   		    
OOHint = -2.3411

GCN = 7.5

delEads_OH = OHslope * GCN + OHint
delEads_OOH = OOHslope * GCN + OOHint
    
'''
Compute ORR rate from *OH and *OOH binding energies
Follows the method and data in S.1 of F. Calle-Vallejo et al., Science 350(6257), 185 (2015).
Also see J. K. Nørskov et al., The Journal of Physical Chemistry B 108(46), 17886 (2004).
for method on how to convert binding energies to activity
'''        

kB = 8.617e-5                      # eV / K
T = 298.15                         # K
U_0 = 1.23                         # eV, theoretical maximum cell voltage for ORR
i_c = 6.1070e-07                    # miliAmperes (mA) per atom, divide by surface area to get current density      
    
# *OH, *OOH
E_g = [-7.53, -13.26]
ZPE = [0.332, 0.428]                # find actual values, eV
TS = [0, 0]                  # eV, at 298 K
E_solv = [-0.575, -0.480]  # solvation energy, eV

# Species free energies at T = 298K
G_OH = E_g[0] + delEads_OH + ZPE[0] - TS[0] + E_solv[0]
#G_O = ???
G_OOH = E_g[1] + delEads_OOH + ZPE[1] - TS[1] + E_solv[1]
G_H2g = -6.8935
G_H2Ol = -14.3182
G_O2g = -9.9294            # gas thermo obtained from ../DFT_data/Volcano_rederive.m

# Reference energies to H2 and O2
G_OH = G_OH - 0.5 * G_H2g - 0.5 * G_O2g
G_OOH = G_OOH - 0.5 * G_H2g - G_O2g
G_H2Ol = G_H2Ol - G_H2g - 0.5 * G_O2g

print G_OH
print G_OOH
print G_H2Ol

print G_OH + G_H2Ol
print 2 * G_H2Ol

print (G_OOH + G_OH +G_H2Ol) / 2 -G_H2Ol