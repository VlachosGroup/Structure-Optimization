'''
Regress a neural network to active site local environment rather than structures
'''

import os
import numpy as np

import matplotlib as mat
import matplotlib.pyplot as plt

def build_KMC_input(cat, fldr_name, trajectory = None):

    '''
    Generate KMC input files for a new structure
    
    :param cat: Structure object
    :param i: Index of the structure
    :param parent_fldr: ./KMC_DB
    '''
    
    if not os.path.exists(fldr_name):
        os.makedirs(fldr_name)
    
    # Show the structure you have generated in the KMC folder
    cat.show(fname = os.path.join(fldr_name,'structure'), fmat = 'png', chop_top = True)
    cat.show(fname = os.path.join(fldr_name,'structure'), fmat = 'xsd', chop_top = False)
    
    # Move non-lattice KMC input files to the folder
    os.system('cp KMC_input/simulation_input.dat ' + fldr_name)
    os.system('cp KMC_input/mechanism_input.dat ' + fldr_name)
    os.system('cp KMC_input/energetics_input.dat ' + fldr_name)
    
    # Generate lattice input file
    cat.graph_to_KMClattice()
    cat.KMC_lat.Write_lattice_input(fldr_name)
    kmc_lat = cat.KMC_lat.PlotLattice()
    kmc_lat.savefig(os.path.join(fldr_name,'lattice.png'),format='png', dpi=600)
    kmc_lat.close()
        
        
    np.save(os.path.join(fldr_name, 'X.npy'), np.array(cat.variable_occs))
    
    '''
    Plot optimization trajectoy
    '''
    
    if not trajectory is None:
        
        plt.figure()
        plt.plot(trajectory[0,:], trajectory[1,:], '-')
        
        plt.xticks(size=18)
        plt.yticks(size=18)
        plt.xlabel('Metropolis step', size=24)
        plt.ylabel('Structure rate', size=24)
        plt.xlim([trajectory[0,0], trajectory[0,-1]])
        plt.ylim([0, None])
        plt.tight_layout()
        plt.savefig(os.path.join(fldr_name, 'trajectory.png'), dpi = 600)
        plt.close()