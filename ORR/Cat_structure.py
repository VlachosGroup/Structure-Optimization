# -*- coding: utf-8 -*-
"""
Created on Fri May 26 13:41:45 2017

@author: mpnun
"""

import os
import numpy as np
import copy
import random
import matplotlib.pyplot as plt
import matplotlib as mat
import matplotlib.ticker as mtick

from ase.build import fcc111, fcc100
from ase.neighborlist import NeighborList
from ase.io import write

from metal import metal
from ORR import ORR_rate
from graph_theory import Graph
from Genetic import MOGA, MOGA_individual

class cat_structure(MOGA_individual):
    
    '''
    Catalyst structure with defects
    '''    
    
    def __init__(self, met_name, facet, dim1, dim2):
        
        self.metal = None
        self.atoms_obj_template = None
        self.active_atoms = None
        self.variable_atoms = None
        self.template_graph = None
        self.defected_graph = None
        self.surface_area = None                 # surface area of the slab in square angstroms
        
        self.evaluated = False
        self.current_density = None
        self.surf_eng = None
        
        '''
        Build the slab
        '''        
        
        self.metal = metal(met_name)
        
        if facet == '111' or facet == 111:
            self.atoms_obj_template = fcc111(met_name, size=(dim1, dim2, 4), vacuum=15.0)
        elif facet == '100' or facet == 100:
            self.atoms_obj_template = fcc100(met_name, size=(dim1, dim2, 4), vacuum=15.0)
        else:
            raise ValueError(str(facet) + ' is not a valid facet.')
            
        self.atoms_obj_template.set_pbc([True, True, False])
        
        # Find neighbors based on distances
        rad_list = ( 2.77 + 0.2 ) / 2 * np.ones(len(self.atoms_obj_template))               # list of neighboradii for each site
        neighb_list = NeighborList(rad_list, self_interaction = False)      # set bothways = True to include both ways
        neighb_list.build(self.atoms_obj_template)
        
        self.template_graph = Graph()
        for i in range(len(self.atoms_obj_template)):
            self.template_graph.add_vertex(i)
        
        for i in range(len(self.atoms_obj_template)):
            for j in neighb_list.neighbors[i]:
                self.template_graph.add_edge([i,j])
            
        self.active_atoms = range(2 * dim1 * dim2, 4 * dim1 * dim2)
        self.variable_atoms = range(3 * dim1 * dim2, 4 * dim1 * dim2)
        self.defected_graph = copy.deepcopy(self.template_graph)        
        
        # Compute surface area for use in normalization      
        self.surface_area = np.linalg.norm( np.cross( self.atoms_obj_template.get_cell()[0,:], self.atoms_obj_template.get_cell()[1,:] ) )        

    
    def randomize(self, coverage = 0.5):
        
        '''
        Randomize the occupancies in the top layer
        '''
        
        n_vacancies = round( coverage * len(self.variable_atoms) )
        n_vacancies = int(n_vacancies)
        vacant_sites = random.sample(self.variable_atoms, n_vacancies)
        for i in vacant_sites:
            self.defected_graph.remove_vertex(i)
        self.evaluated = False
        
    
    def get_OFs(self):
        
        '''
        Evaluate the objective functions
        '''
        
        if not self.evaluated:          # Avoid repeated evaluations if the structure has not changed
            self.current_density = self.eval_current_density()
            self.surf_eng = self.eval_surface_energy()
            self.evaluated = True
        
        return [self.surf_eng, -self.current_density]
    
    
    def mutate(self):
        
        '''
        Mutates an individual to yield an offspring
        Used in the genetic algorithm
        '''
        
        child = copy.deepcopy(self)
        atom_to_flip = random.choice(child.variable_atoms)
        child.flip_atom(atom_to_flip)
        child.evaluated = False
        return child
        
    
    def crossover(self, mate):
        
        '''
        Crossover with a mate
        Return the child
        '''
        
        child = copy.deepcopy(self)
        child.defected_graph = copy.deepcopy(child.template_graph)
        
        ind = 0
        for site in child.variable_atoms:
            
            if ind < len(child.variable_atoms) / 2:
                parent = self
            else:
                parent = mate

            if not parent.defected_graph.is_node(site):
                child.defected_graph.remove_vertex(site)
        
            ind += 1
        
        child.evaluated = False
        return child
    
    
    def eval_current_density(self, atom_graph = None, normalize = True):
        
        '''
        Normalized: current density [mA/cm^2]
        Not normalized: current [mA]
        '''

        if atom_graph is None:
            atom_graph = self.defected_graph

        curr = 0
        for i in self.active_atoms:
            if atom_graph.is_node(i):
                if atom_graph.get_coordination_number(i) <= 9:
                    gcn = atom_graph.get_generalized_coordination_number(i, 12)
                    BE_OH = self.metal.get_OH_BE(gcn)
                    BE_OOH = self.metal.get_OOH_BE(gcn)
                    curr += ORR_rate(BE_OH, BE_OOH)
                
        if normalize:
            curr = curr / self.surface_area           # normalize by surface area
            
        return curr
        
    
    def eval_surface_energy(self, atom_graph = None, normalize = True):
        
        '''
        Normalized: surface energy [J/m^2]
        Not normalized: formation energy [eV]
        '''        
        
        if atom_graph is None:
            atom_graph = self.defected_graph
        
        E_form = 0
        for i in self.active_atoms:
            if atom_graph.is_node(i):
                E_form += self.metal.E_coh * ( 1 - np.sqrt( atom_graph.get_coordination_number(i) / 12.0 ) )
                
        if normalize:
            E_form = E_form / self.surface_area           # normalize by surface area
                
        return E_form

        
    def get_defected_mols(self):
        
        '''
        Use defected graph and template atoms object to generate
        the atoms object for the defected structure
        '''
        
        atoms_obj = copy.deepcopy(self.atoms_obj_template)
        delete_these = [False for i in range(len(atoms_obj))]
        for i in self.variable_atoms:
            if not self.defected_graph.is_node(i):
                delete_these[i] = True
        
        delete_these = np.array(delete_these)
        del atoms_obj[delete_these]        
        return atoms_obj
        
    
    def flip_atom(self, ind):
        
        '''
        If atom number ind is present in the defected graph, remove it.
        If it is not present, add it and all edges to adjacent atoms.
        '''        
        
        if self.defected_graph.is_node(ind):
            self.defected_graph.remove_vertex(ind)
        else:
            self.defected_graph.add_vertex(ind)
            for neighb in self.template_graph.get_neighbors(ind):
                if self.defected_graph.is_node(neighb):
                    self.defected_graph.add_edge([ind, neighb])
    

    def show(self, gen_num):
                
        '''
        Print image of surface
        '''        
        
        defect_atoms_obj = self.get_defected_mols()
        
        coords = defect_atoms_obj.get_positions()
        a_nums = defect_atoms_obj.get_atomic_numbers()
        chem_symbs = defect_atoms_obj.get_chemical_symbols()
        
        # Change top layer atoms to Ni
        top_layer_z = np.max(coords[:,2])
        for atom_ind in range(len(defect_atoms_obj)):
            if coords[atom_ind,2] > top_layer_z - 0.1:
                a_nums[atom_ind] = 27
                chem_symbs[atom_ind] = 'Co'
                
        defect_atoms_obj.set_atomic_numbers(a_nums)
        defect_atoms_obj.set_chemical_symbols(chem_symbs)
        
        write('best_struc_' + str(gen_num) + '.png', defect_atoms_obj )
        
        
if __name__ == "__main__":
    
    os.system('clear')
        
    # Numerical parameters
    p_count = 100                   # population size    
    n_gens = 1000                    # number of generations
    
    x = MOGA()
    x.P = [cat_structure('Pt', '111', 8, 8) for i in range(p_count)]
    x.randomize_pop()
    x.genetic_algorithm(n_gens, n_snaps = 100)