'''
Test evlaution of the linear site coupling model on some test structures
'''

'''
Show the local environment of a site
'''

import sys
sys.path.append('/home/vlachos/mpnunez/Github/Zacros-Wrapper')
sys.path.append('/home/vlachos/mpnunez/Github/Structure-Optimization')
sys.path.append('/home/vlachos/mpnunez/python_packages/ase')
sys.path.append('/home/vlachos/mpnunez/python_packages/sklearn/lib/python2.7/site-packages')

import os
import pickle
import numpy as np
import matplotlib as mat
import matplotlib.pyplot as plt
from multiprocessing import Pool
from mpi4py import MPI      # MPI parallelization

import zacros_wrapper as zw
import OML

# Make a random structure
cat = OML.toy_cat()
cat.randomize(coverage = 0.7, build_structure = True)
cat.show(fname = 'test_struc')
cat.graph_to_KMClattice()

kmc_site_rates = OML.compute_site_rates_lsc(cat.KMC_lat)
site_rates = np.zeros(len(cat.variable_occs))
for i in range(len(kmc_site_rates)):
    site_rates[cat.var_ind_kmc_sites[i]] = kmc_site_rates[i]
    
print kmc_site_rates
print site_rates
print len(kmc_site_rates)
print len(site_rates)