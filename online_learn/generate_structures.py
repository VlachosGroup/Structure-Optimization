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

import random
import time
from multiprocessing import Pool

from NH3.NiPt_NH3 import NiPt_NH3
from KMC_handler import *

'''
Generate data set
'''



def square(i):

    np.random.seed(i)

    n_strucs = 50
    
    cat = NiPt_NH3()
    DB_fldr = '/home/vlachos/mpnunez/OML_data/NH3_data_2/KMC_DB'
    kmc_src = '/home/vlachos/mpnunez/OML_data/NH3_data_2/KMC_input'
    fldr_name = os.path.join(DB_fldr, 'structure_' + str(i+1) )
    print fldr_name
    
    
    '''
    Randomize the coverage
    '''
    #cat.randomize(coverage = 0.1 + 0.8 * float(i) / (n_strucs+1), build_structure = False)
    
    '''
    Make islands
    '''
    
    cat.variable_occs = [0 for j in range(cat.atoms_per_layer)]
    n_tops = int( (3 + 15 * float(i) / (n_strucs+1)))
    top_sites = np.random.choice( range(len(cat.variable_atoms)), size=n_tops, replace=False )
    
    for i in top_sites:     # Pick a site to build an island around
        
        sym_inds = cat.var_ind_to_sym_inds(i)
        
        ind1 = cat.sym_inds_to_var_ind(sym_inds[0] - 2, sym_inds[1] + 2)
        cat.variable_occs[ind1] = 1
        ind1 = cat.sym_inds_to_var_ind(sym_inds[0] - 1, sym_inds[1] + 2)
        cat.variable_occs[ind1] = 1
        ind1 = cat.sym_inds_to_var_ind(sym_inds[0] , sym_inds[1] + 2)
        cat.variable_occs[ind1] = 1
        
        ind1 = cat.sym_inds_to_var_ind(sym_inds[0] - 2, sym_inds[1] + 1)
        cat.variable_occs[ind1] = 1
        ind1 = cat.sym_inds_to_var_ind(sym_inds[0] - 1, sym_inds[1] + 1)
        cat.variable_occs[ind1] = 1
        ind1 = cat.sym_inds_to_var_ind(sym_inds[0] , sym_inds[1] + 1)
        cat.variable_occs[ind1] = 1
        ind1 = cat.sym_inds_to_var_ind(sym_inds[0] +1, sym_inds[1] + 1)
        cat.variable_occs[ind1] = 1
        
        ind1 = cat.sym_inds_to_var_ind(sym_inds[0] - 2, sym_inds[1])
        cat.variable_occs[ind1] = 1
        ind1 = cat.sym_inds_to_var_ind(sym_inds[0] - 1, sym_inds[1])
        cat.variable_occs[ind1] = 1
        ind1 = cat.sym_inds_to_var_ind(sym_inds[0] , sym_inds[1] )
        cat.variable_occs[ind1] = 1
        ind1 = cat.sym_inds_to_var_ind(sym_inds[0] +1, sym_inds[1])
        cat.variable_occs[ind1] = 1
        ind1 = cat.sym_inds_to_var_ind(sym_inds[0] +2, sym_inds[1])
        cat.variable_occs[ind1] = 1
        
        ind1 = cat.sym_inds_to_var_ind(sym_inds[0] - 1, sym_inds[1] - 1)
        cat.variable_occs[ind1] = 1
        ind1 = cat.sym_inds_to_var_ind(sym_inds[0] , sym_inds[1] - 1)
        cat.variable_occs[ind1] = 1
        ind1 = cat.sym_inds_to_var_ind(sym_inds[0] +1, sym_inds[1] - 1)
        cat.variable_occs[ind1] = 1
        ind1 = cat.sym_inds_to_var_ind(sym_inds[0] +2, sym_inds[1] - 1)
        cat.variable_occs[ind1] = 1
        
        ind1 = cat.sym_inds_to_var_ind(sym_inds[0] , sym_inds[1] - 2)
        cat.variable_occs[ind1] = 1
        ind1 = cat.sym_inds_to_var_ind(sym_inds[0] + 1, sym_inds[1] - 2)
        cat.variable_occs[ind1] = 1
        ind1 = cat.sym_inds_to_var_ind(sym_inds[0] +2, sym_inds[1] - 2)
        cat.variable_occs[ind1] = 1
    
    cat.occs_to_atoms()
    cat.occs_to_graph()
    cat.graph_to_KMClattice()
    
    # Build input files
    write_structure_files(cat, fldr_name)
    
if __name__ == '__main__': 
    
    square(1)
    
    ## Run in parallel
    #pool = Pool()
    #pool.map(square, range(0,16))        # change 1 to 96
    #pool.close()