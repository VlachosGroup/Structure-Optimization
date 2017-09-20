'''
Adds optimization functionality to the ORR catalyst structure
'''

import random
import numpy as np

import sys
sys.path.append('/home/vlachos/mpnunez/Github/Structure-Optimization/ORR/')
from Cat_structure import cat_structure


class cat_optimize(cat_structure):
    
    '''
    Inherits ORR catalyst structure and adds optimization functionality
    '''
    
    Pt_Pt_1nn_dist = 2.77       # angstrom
    
    def __init__(self):
        
        '''
        Call superclass constructor
        '''
        
        #super(cat_optimize, self).__init__(met_name = 'Pt', facet = 111, dim1 = 12, dim2 = 12)
        cat_structure.__init__(self, met_name = 'Pt', facet = 111, dim1 = 12, dim2 = 12)
        self.atom_last_moved = None
        self.weights = [0., -1.]
        
        
    def geo_crossover(self, x1, x2, pt1 = 1, pt2 = 1):
        '''
        Geometry-based crossover. Partions the catalyst surface into regions in a checkerboard style and performs crossover.
        
        :param x1: Mom
        :param x2: Dad
        :param pt1: By default, use 1-point crossover in first dimension
        :param pt2: By default, use 1-point crossover in second dimension
        
        :returns: Two offspring
        '''
        
        x_bounds = [random.random() for i in xrange(pt1)]
        y_bounds = [random.random() for i in xrange(pt2)]
        
        frac_coords = self.atoms_obj_template.get_scaled_positions()
        
        for i in xrange(len(x1)):
            
            # Find whether it is an even or odd cell
            score = 0
            for bound in x_bounds:
                if frac_coords[self.variable_atoms[i],0] > bound:
                    score += 1
            for bound in y_bounds:
                if frac_coords[self.variable_atoms[i],1] > bound:
                    score += 1
            
            # Swap if it is in an even cell
            if score % 2 == 0:
                x1[i], x2[i] = x2[i], x1[i]
        
        return x1, x2
    
    
    def eval_x(self, x):
    
        x = np.array(x)
        self.evaluated = False
        # Build the defected graph
        self.defected_graph = self.template_graph.copy_data()
        for i in range(len(x)):
            if x[i] == 0:
                self.defected_graph.remove_vertex(self.variable_atoms[i])
        
        return self.get_OFs()
        
    
    def get_OF(self):
        
        '''
        :param weights: Weights for the objective functions
        :returns: A single objective function that is a linear superposition of the other objectives
        '''

        return self.weights[0] * self.eval_surface_energy() + self.weights[1] * self.eval_current_density()
        
    
    def get_OFs(self):
        
        '''
        Evaluate the objective functions
        :returns: 2-ple of surface energy and current density
        '''
        
        return self.eval_surface_energy(), self.eval_current_density() #+ np.random.normal(loc=0.0, scale=0.025)
     
        
    def rand_move(self):
        self.atom_last_moved = random.choice(self.variable_atoms)
        self.flip_atom(self.atom_last_moved)
    
    def revert_last(self):
        self.flip_atom(self.atom_last_moved)
    