'''
Test simulated annealing using the linear site coupling analytical evaluation
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
import random

import zacros_wrapper as zw
import OML

n_seed = int(sys.argv[1])
random.seed(a=n_seed)
site_rates_DB = np.load('site_rates_DB.npy')

# Make a random structure
cat = OML.LSC_cat()
cat.randomize(coverage = 0.7, build_structure = True)
cat.show(fname = 'before_optimization_' + sys.argv[1])

#cat.eval_struc_rate_anal(cat.variable_occs)
#raise NameError('stop')

OML.optimize(cat,mode = 'analytical', n_cycles = 100, T_0 = np.max(site_rates_DB), fldr = '.', prefix = sys.argv[1]+'_')
cat.show(fname = 'after_optimization_' + sys.argv[1])
cat.show(fname = 'after_optimization_' + sys.argv[1],fmat = 'xsd')