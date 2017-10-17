'''
Regress a neural network to active site local environment rather than structures
'''

import sys
sys.path.append('/home/vlachos/mpnunez/Github/Zacros-Wrapper')
sys.path.append('/home/vlachos/mpnunez/Github/Structure-Optimization')
sys.path.append('/home/vlachos/mpnunez/python_packages/ase')
sys.path.append('/home/vlachos/mpnunez/python_packages/sklearn/lib/python2.7/site-packages')
import os

import numpy as np

from cat_optimization import *
from NH3.NiPt_NH3_simple import NiPt_NH3_simple

import random
import time
from multiprocessing import Pool

'''
Generate data set
'''



def square(i):

    np.random.seed(i)

    n_strucs = 96
    cat = NiPt_NH3_simple()
    
    fldr_name = os.path.join( '/home/vlachos/mpnunez/NN_data/AB_data/KMC_DB', 'structure_' + str(i+1) )
    n_tops = int( (10 + 10 * float(i) / (n_strucs+1)))
    
    
    cat = NiPt_NH3_simple()
    #cat.randomize(coverage = 0.7, build_structure = True)
    cat.variable_occs = [0 for i in range(cat.atoms_per_layer)]
    cat.occs_to_atoms()
    
    if not os.path.exists(fldr_name):
        os.makedirs(fldr_name)
    
    top_sites = np.random.choice( range(len(cat.variable_atoms)), size=n_tops, replace=False )
    
    for i in top_sites:
        
        cat.variable_occs[i] = 1
        sym_inds = cat.var_ind_to_sym_inds(i)
        
        ind1 = cat.sym_inds_to_var_ind(sym_inds[0] - 1, sym_inds[1] + 1)
        cat.variable_occs[ind1] = 1
        ind1 = cat.sym_inds_to_var_ind(sym_inds[0], sym_inds[1] + 1)
        cat.variable_occs[ind1] = 1
        ind1 = cat.sym_inds_to_var_ind(sym_inds[0] - 1, sym_inds[1])
        cat.variable_occs[ind1] = 1
        ind1 = cat.sym_inds_to_var_ind(sym_inds[0] + 1, sym_inds[1])
        cat.variable_occs[ind1] = 1
        ind1 = cat.sym_inds_to_var_ind(sym_inds[0], sym_inds[1] - 1)
        cat.variable_occs[ind1] = 1
        ind1 = cat.sym_inds_to_var_ind(sym_inds[0] + 1, sym_inds[1] - 1)
        cat.variable_occs[ind1] = 1
    
    cat.occs_to_atoms()
    cat.occs_to_graph()
    
    
    cat.show(fname = os.path.join(fldr_name,'structure'), fmat = 'png', chop_top = True)
    cat.show(fname = os.path.join(fldr_name,'structure'), fmat = 'xsd', chop_top = False)

    os.system('cp /home/vlachos/mpnunez/NN_data/AB_data/KMC_input/simulation_input.dat ' + fldr_name)
    os.system('cp /home/vlachos/mpnunez/NN_data/AB_data/KMC_input/mechanism_input.dat ' + fldr_name)
    os.system('cp /home/vlachos/mpnunez/NN_data/AB_data/KMC_input/energetics_input.dat ' + fldr_name)
    
    cat.graph_to_KMClattice()
    cat.KMC_lat.Write_lattice_input(fldr_name)
    kmc_lat = cat.KMC_lat.PlotLattice()
    kmc_lat.savefig(os.path.join(fldr_name,'lattice.png'),format='png', dpi=1000)
    kmc_lat.close()
    
    print fldr_name
    return cat.variable_occs
    
if __name__ == '__main__': 
    
    # Run in parallel
    pool = Pool()
    y_vec = pool.map(square, range(96))        # change 1 to 96
    pool.close()
    
    np.save('X.npy', np.array(y_vec))