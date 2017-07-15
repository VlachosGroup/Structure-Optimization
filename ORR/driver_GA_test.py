# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 17:31:22 2017

@author: mpnun
"""

import numpy as np
import os
from Genetic import MOGA


class GA_ind():
    
    def __init__(self):
        
        self.x = None                     # List of individuals


    def randomize(self, coverage = 0.5):
        
        a = -10.
        b = 10.
        self.x = (b - a) * np.random.random() + a
        
    
    def eval_OFs(self):
        
        '''
        Evaluate the objective functions
        '''
        
        return [self.x ** 2, (self.x - 2) **2 ]
    
    
    def mutate(self, omega = 1.):
        
        self.x = self.x + omega * np.random.randn()
        
    
    def crossover(self, mate):
        child = GA_ind()
        child.x = np.mean([self.x, mate.x])
        return child
        
if __name__ == "__main__":
    


    os.system('clear')
        
    # Numerical parameters
    p_count = 100                   # population size    
    n_gens = 1000                    # number of generations
    
    x = MOGA()
    x.pop = [GA_ind() for i in range(p_count)]
    x.randomize_pop()
    x.genetic_algorithm(n_gens, n_snaps = 10)