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
from build_KMC_input import *

import random
import time
from multiprocessing import Pool

'''
Generate data set
'''



def square(i):

    np.random.seed(i)

    n_strucs = 50
    
    cat = NiPt_NH3_simple()
    fldr_name = os.path.join( '/home/vlachos/mpnunez/NN_data/AB_data_4/KMC_DB', 'structure_' + str(i+1) )
    print fldr_name
    cat.randomize(coverage = 0.1 + 0.8 * float(i) / (n_strucs+1), build_structure = False)
    
    #cat.variable_occs = [0 for i in range(cat.atoms_per_layer)]
    #n_tops = int( (10 + 10 * float(i) / (n_strucs+1)))
    #top_sites = np.random.choice( range(len(cat.variable_atoms)), size=n_tops, replace=False )
    #
    #for i in top_sites:
    #    
    #    cat.variable_occs[i] = 1
    #    sym_inds = cat.var_ind_to_sym_inds(i)
    #    
    #    ind1 = cat.sym_inds_to_var_ind(sym_inds[0] - 1, sym_inds[1] + 1)
    #    cat.variable_occs[ind1] = 1
    #    ind1 = cat.sym_inds_to_var_ind(sym_inds[0], sym_inds[1] + 1)
    #    cat.variable_occs[ind1] = 1
    #    ind1 = cat.sym_inds_to_var_ind(sym_inds[0] - 1, sym_inds[1])
    #    cat.variable_occs[ind1] = 1
    #    ind1 = cat.sym_inds_to_var_ind(sym_inds[0] + 1, sym_inds[1])
    #    cat.variable_occs[ind1] = 1
    #    ind1 = cat.sym_inds_to_var_ind(sym_inds[0], sym_inds[1] - 1)
    #    cat.variable_occs[ind1] = 1
    #    ind1 = cat.sym_inds_to_var_ind(sym_inds[0] + 1, sym_inds[1] - 1)
    #    cat.variable_occs[ind1] = 1
    
    cat.occs_to_atoms()
    cat.occs_to_graph()
    
    # Build input files
    build_KMC_input(cat, fldr_name, trajectory = None)
    
    
if __name__ == '__main__': 
    
    # Run in parallel
    pool = Pool()
    pool.map(square, range(0,50))        # change 1 to 96
    pool.close()