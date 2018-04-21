# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 13:48:34 2016

@author: mpnun
"""

import numpy as np
import matplotlib as mat
import matplotlib.pyplot as plt

import zacros_wrapper as zw
from multiprocessing import Pool
import os

def square(i):

    RunPath = 'structure_' + str(i+1)
    print RunPath
    
    y = zw.kmc_traj()
    y.Path = RunPath
    #y.ReadAllInput()
    #y.exe_file = exe_file
    y.ReadAllOutput(build_lattice=True)
    
    
    y.PlotSurfSpecVsTime(site_norm = 1)
    y.PlotGasSpecVsTime()
    y.PlotElemStepFreqs(site_norm = 1, time_norm = False)
    
    
    '''
    Plot heat maps
    '''
    
    n_sites = len( y.lat.site_type_inds )
    
    cutoff = 3.0
    type_symbols = ['o','s','^','v', '<', '>', '8', 
            'd', 'D', 'H', 'h', '*', 'p', '+', 
            ',', '.', '1', '2', '3', '4', '_', 'x', 
            '|', 0, 1, 10, 11, 2, 3, 4, 5, 6, 7, 8]
    ms = 4
            
    
    border = np.dot(np.array([[0.0,0.0],[1.0,0.0],[1.0,1.0],[0.0,1.0],[0.0,0.0]]), y.lat.lattice_matrix)             
    
    mat.rcParams['mathtext.default'] = 'regular'
    mat.rcParams['text.latex.unicode'] = 'False'
    mat.rcParams['legend.numpoints'] = 1
    mat.rcParams['lines.linewidth'] = 2
    mat.rcParams['lines.markersize'] = 12
    
    
    for reaction in range(len(y.genout.RxnNameList)):
    
        rxn_name = y.genout.RxnNameList[reaction]
    
        plt.figure()
        
        plt.plot(border[:,0], border[:,1], '--k', linewidth = 2)                  # cell border 
        
        # Plot sites
        for site_type in range(1, np.max(np.array(y.lat.site_type_inds))+1 ):
        
            is_of_type = []
            
            for site_ind in range(len(y.lat.site_type_inds)):
                if y.lat.site_type_inds[site_ind] == site_type:
                    is_of_type.append(site_ind)
            
            if site_type == 1:
                plt.plot(y.lat.cart_coords[is_of_type,0], y.lat.cart_coords[is_of_type,1], linestyle='None', marker = 'o', color = 'b', markersize = 9)          # sites  
            elif site_type == 2:
                plt.plot(y.lat.cart_coords[is_of_type,0], y.lat.cart_coords[is_of_type,1], linestyle='None', marker = 'o', color = 'g', markersize = 9)          # sites 
            elif site_type == 3:
                plt.plot(y.lat.cart_coords[is_of_type,0], y.lat.cart_coords[is_of_type,1], linestyle='None', marker = 'o', color = 'm', markersize = 9)          # sites 
                
        # Plot active sites
        
            is_active = []
            for site_ind in range(len(y.lat.site_type_inds)):
                
                if y.TS_site_props_ss[site_ind,reaction] > 0:
                    is_active.append(site_ind)
    
            plt.plot(y.lat.cart_coords[is_active,0], y.lat.cart_coords[is_active,1], linestyle='None', marker = '^', color = 'r', markersize = ms)          # sites  
        
        # Choose range to plot
        xmin = np.min(border[:,0])
        xmax = np.max(border[:,0])
        delx = xmax - xmin
        ymin = np.min(border[:,1])
        ymax = np.max(border[:,1])
        dely = ymax - ymin
        mag = 0.1        
        
        plt.title(rxn_name, size = 20)
        plt.xlim([xmin - mag * delx, xmax + mag * delx])
        plt.ylim([ymin - mag * dely, ymax + mag * dely])
        plt.xticks(size=20)
        plt.yticks(size=20)
        plt.xlabel('x-coord (ang)',size=24)
        plt.ylabel('y-coord (ang)',size=24)
        plt.axis('equal')
        plt.tight_layout()
        
        plt.savefig( os.path.join(RunPath, 'Heat_map_' + rxn_name + '.png' ) )
        plt.close()
        
    return None
        
if __name__ == '__main__': 
    
    # Run in parallel
    pool = Pool()
    y_vec = pool.map(square, range(96))        # change 1 to 96
    pool.close()