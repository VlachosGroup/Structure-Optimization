# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 17:31:22 2017

@author: mpnun
"""

import numpy as np
import random
import os
import copy
from Genetic import MOGA, SOGA, Bifidelity
        
if __name__ == "__main__":
    
    os.system('clear')
    
    random.seed(a=12345)
    np.random.seed(seed=12345)
    os.chdir('/home/vlachos/mpnunez/Optimization_data/ORR_GA')
    
    '''
    Genetic algorithm optimization
    '''
    
    # Numerical parameters
    p_count = 208                   # population size, make a multiple of the number of cores ( = 16), use 208
    n_gens = 10000                    # number of generations, can restart if more generations are needed
    
    #ga = SOGA()
    ga = MOGA()
    ga = Bifidelity()
    ga.eval_obj = cat_structure(met_name = 'Pt', facet = '111', dim1 = 12, dim2 = 12)
    ga.P = np.array([ga.eval_obj.randomize() for i in range(p_count)])
    ga.genetic_algorithm(n_gens, n_snaps = 101)