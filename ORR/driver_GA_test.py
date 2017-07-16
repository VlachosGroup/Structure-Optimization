# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 17:31:22 2017

@author: mpnun
"""

import numpy as np
import os
from Genetic import MOGA, MOGA_individual
import copy

class GA_ind(MOGA_individual):
    
    def __init__(self):
        
        self.x = None
        self.OFs = None


    def randomize(self):
        
        a = -10.
        b = 10.
        self.x = (b - a) * np.random.random() + a
        
    
    def get_OFs(self):
        
        '''
        Return the objective function values and evaluate them if necessary
        '''
        
        if self.OFs is None:
            self.eval_OFs()
            
        return self.OFs
    
    def eval_OFs(self):
        
        '''
        Evaluate the objective functions
        '''
        
        self.OFs = [self.x ** 2, (self.x - 2) **2 ]
    
    
    def mutate(self, omega = 1.):
        
        '''
        Return a mutated individual
        '''
        
        child = copy.deepcopy(self)
        child.x = child.x + omega * np.random.randn()
        child.OFs = None
        return child
        
    
    def crossover(self, mate):
        
        '''
        Return a crossover child from two parents
        '''
        
        child = GA_ind()
        child.x = np.mean([self.x, mate.x])
        return child
        
if __name__ == "__main__":
    


    os.system('clear')
        
    # Numerical parameters
    p_count = 100                   # population size    
    n_gens = 1000                    # number of generations
    
    x = MOGA()
    x.P = [GA_ind() for i in range(p_count)]
    x.randomize_pop()
    x.genetic_algorithm(n_gens, n_snaps = 10)