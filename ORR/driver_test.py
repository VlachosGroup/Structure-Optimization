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
from Cat_structure import cat_structure
      

if __name__ == "__main__":
    
    os.system('clear')
    
    random.seed(a=12345)
    np.random.seed(seed=12345)
    os.chdir('C:\Users\mpnun\Desktop')

    '''
    Genetic algorithm optimization
    '''
    
    # Numerical parameters
    p_count = 208                   # population size, make a multiple of the number of cores ( = 16), use 208
    n_gens = 1000                    # number of generations, can restart if more generations are needed
    
    #ga = SOGA()
    #ga = MOGA()
    eval_obj = cat_structure(met_name = 'Pt', facet = '111', dim1 = 6, dim2 = 6)
    x = eval_obj.randomize(coverage = 0.1)
    eval_obj.eval_x(x)
    eval_obj.show()
    eval_obj.get_Nnn()