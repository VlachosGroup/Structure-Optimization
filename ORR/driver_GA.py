# -*- coding: utf-8 -*-
"""
Created on Wed Jul 05 12:16:07 2017

@author: mpnun
"""

import os

from Genetic import MOGA
from Cat_structure import cat_structure

os.system('clear')
    
# Numerical parameters
p_count = 10                   # population size    
n_gens = 100                    # number of generations

x = MOGA()
x.pop = [cat_structure('Pt', '111', 8, 8) for i in range(p_count)]
x.randomize_pop()
x.genetic_algorithm(n_gens, n_snaps = 10)