# -*- coding: utf-8 -*-
"""
Created on Fri May 26 14:39:12 2017

@author: mpnun
"""

import numpy as np

def ORR_rate(delEads_OH, delEads_OOH):
    
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
    G_OOH = E_g[1] + delEads_OOH + ZPE[1] - TS[1] + E_solv[1]
    G_H2g = -6.8935
    G_H2Ol = -14.3182
    G_O2g = -9.9294            # gas thermo obtained from ../DFT_data/Volcano_rederive.m
    
    # Compute G1 and G4
    G1 = G_OOH - G_O2g - 0.5 * G_H2g
    G4 = G_H2Ol - G_OH - 0.5 * G_H2g
    
    n = 1          # number of electrons transferred each step
    U1 = -G1/n
    U4 = -G4/n
    U = min(U1,U4)     # minimum potential you can run the cell at so that no step is activated (onset potential?)
    
    j = i_c * np.exp( - (U_0 - U) / (kB * T) )
    
    # Check which step is rate determining
    is_step_4 = G1 < G4;          # *OH desorption (Step 4) is rate determining
    if is_step_4:
        RDS = 4
    else:
        RDS = 1
    
    return j