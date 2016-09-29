# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 14:22:15 2016

@author: mpnun
"""

import numpy as np

from Cat_structure import Cat_structure

class Machine_learning:
    
    def __init__(self):
        
        self.population = []
        self.archive = []
        self.data = []
        
    def eval_pop(self):
        for struc in self.population:
            struc.evaluate_eng()        
    
    def Bay_prior(self,struc):
        return 1
        
    def Bay_post(self,struc):
        return 1.0 * self.Bay_prior(struc)