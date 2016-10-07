# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 14:22:15 2016

@author: mpnun
"""

import numpy as np
import copy
import matplotlib.pyplot as plt
import matplotlib as mat

from Cat_structure import Cat_structure
from Cat_structure import Pattern

class Machine_learning:
    
    def __init__(self):
        
        self.cat_structure = Cat_structure()
        self.data = []
        self.patterns = []
        self.pattern_engs = []
        
        # Use later
        self.population = []            # for genetic algorithm
        self.archive = []        
        
    def eval_structure(self, struc):
        
        eng = 0
        for pair_ind in range(len(self.pattern_engs)):
            eng += self.pattern_engs[pair_ind] * struc.count_pairs(self.patterns[pair_ind][0], self.patterns[pair_ind][1])
        return eng
    
    def Metropolis(self, n_steps = 5000, T = 1.0):
        
        E = self.eval_structure(self.cat_structure)        
        self.data.append(E)         # record energy 
        
        for step in range(n_steps):
            
            new_struc = copy.deepcopy(self.cat_structure)
            new_struc.mutate()
            E_new = self.eval_structure(new_struc)
            
            if np.exp( - (E_new - E) / T ) > np.random.uniform(0, 1):       # Metropolis criterion
                self.cat_structure = new_struc
                E = E_new
            
            self.data.append(E)             # record energy
            
    def PlotTrajectory(self):

        mat.rcParams['mathtext.default'] = 'regular'
        mat.rcParams['text.latex.unicode'] = 'False'
        mat.rcParams['legend.numpoints'] = 1
        mat.rcParams['lines.linewidth'] = 2
        mat.rcParams['lines.markersize'] = 12
                
        plt.figure()
        plt.plot(range(len(self.data)), self.data)    
        plt.xticks(size=20)
        plt.yticks(size=20)
        plt.xlabel('iteration',size=24)
        plt.ylabel('energy',size=24)   
        plt.show()

# Use later '''
    
    def eval_pop(self):
        for struc in self.population:
            struc.evaluate_eng()        
    
	# Relative probability, before data
    def Bay_prior(self,struc):
        return 1
    
	# Relative probability, after data 
    def Bay_post(self,struc):
        return 1.0 * self.Bay_prior(struc)