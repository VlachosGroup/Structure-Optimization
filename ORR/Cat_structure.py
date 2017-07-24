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

import sys
sys.path.append('C:\Users\mpnun\OneDrive\Documents\\ase')
from ase.build import fcc111, fcc100
from ase.neighborlist import NeighborList
from ase.io import read
from ase.io import write

from metal import metal
from ORR import ORR_rate
from graph_theory import Graph
from Genetic import MOGA, MOGA_individual

class cat_structure(MOGA_individual):
    
    '''
    Catalyst structure with defects
    '''
    
    def __init__(self, met_name = None, facet = None, dim1 = None, dim2 = None):
        
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
        
        if not met_name is None:
        
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


    def copy_data(self):
        
        '''
        Create a new individual with identical essential data
        '''
        
        child = cat_structure()
        child.metal = self.metal
        child.active_atoms = self.active_atoms
        child.variable_atoms = self.variable_atoms
        child.template_graph = self.template_graph
        child.defected_graph = self.defected_graph.copy_data()
        child.surface_area = self.surface_area
        child.atoms_obj_template = self.atoms_obj_template
        return child
    
    
    def randomize(self, coverage = None):
        
        '''
        Randomize the occupancies in the top layer
        '''
        
        if coverage is None:
            coverage = random.random()
        n_vacancies = round( coverage * len(self.variable_atoms) )
        n_vacancies = int(n_vacancies)
        vacant_sites = random.sample(self.variable_atoms, n_vacancies)
        for i in vacant_sites:
            self.defected_graph.remove_vertex(i)
        self.evaluated = False
        
    
    def load_from_file(self, ftoread, d_cut = 0.001):
    
        '''
        Determine which atoms in the template are missing in the defected structure
        d_cut: distance cutoff in angstroms
        '''

        ASE_defected = read(ftoread, format = 'xsd')

        for atom_ind in range(len(self.atoms_obj_template)):
        
            # Get position and atomic number of the template atom we are trying to find
            cart_coords = self.atoms_obj_template.get_positions()[atom_ind, :]
            atomic_num = self.atoms_obj_template.get_atomic_numbers()[atom_ind]
            
            defect_ind = 0      # index of defect atom which might be a match
            dist = 1.0      # distance between atoms we are trying to match
            
            match_found = False
            
            while (not match_found) and defect_ind < len(ASE_defected):
            
                defect_coords = ASE_defected.get_positions()[defect_ind, :]
                defect_an = ASE_defected.get_atomic_numbers()[defect_ind]
                dist = np.linalg.norm( cart_coords - defect_coords )
                match_found = (dist < d_cut) #and (defect_an == atomic_num)         # We do not need to check whether the elements match
                defect_ind += 1
                
            if not match_found:
                self.defected_graph.remove_vertex(atom_ind)

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
    

    def mutate(self, sigma):
        
        '''
        Mutates an individual to yield an offspring
        Used in the genetic algorithm
        '''
        
        n_flipped = 0
        for atom_to_flip in self.variable_atoms:
            if np.random.random() < 1.0 / len(self.variable_atoms):
                self.flip_atom(atom_to_flip)
                n_flipped += 1
        
        if n_flipped > 0:
            self.evaluated = False
        
    
    def crossover(self, mate):
        
        '''
        Crossover with a mate
        Return the child
        '''
        
        child = self.copy_data()
        child.defected_graph = child.template_graph.copy_data()
        
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
            curr = curr / ( self.surface_area * 1.0e-16)          # normalize by surface area (in square centimeters)
  
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
            E_form = E_form * 1.60218e-19                                             # convert energy from eV to Joules
            E_form = E_form / ( self.surface_area * 1.0e-20)                # normalize by surface area (in square meters)
                
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
    

    def show(self, n_struc, fmat = 'picture'):
                
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
        
        if fmat == 'picture':
            write('structure_' + str(n_struc) + '.png', defect_atoms_obj )
        elif fmat == 'xsd':
            write('structure_' + str(n_struc) + '.xsd', defect_atoms_obj, format = fmat )

        
if __name__ == "__main__":
    
    os.system('clear')
#    os.chdir('C:\Users\mpnun\Desktop\GA_run')
    os.chdir('C:\Users\mpnun\Desktop\GA_run_multi')
    
    best_xsd = 'C:\Users\mpnun\Desktop\GA_run\most_active.xsd'

    template = cat_structure(met_name = 'Pt', facet = '111', dim1 = 10, dim2 = 10)
    
    active_opt = cat_structure(met_name = 'Pt', facet = '111', dim1 = 10, dim2 = 10)
    active_opt.load_from_file(best_xsd)
#    ofs = active_opt.get_OFs()
#    print 'Surface energy: ' + str(ofs[0]) + ' J/m^2'
#    print 'Current density: ' + str(ofs[1]) + ' mA/cm^2'
#    raise NameError('stop')
    
    
    '''
    Genetic algorithm optimization
    '''
    
    # Numerical parameters
    p_count = 50                   # population size
    n_gens = 1000                    # number of generations
    
    x = MOGA()
    x.P = [template.copy_data() for i in range(p_count)]
    x.randomize_pop()
    x.P[0] = template.copy_data()       # Make sure that the ideal surface is included in the initial population
    x.P[1] = active_opt                 # Put the activity optimum in the initial population as well
    x.genetic_algorithm(n_gens, n_snaps = 11, n_obj = 2)
    
    x.P[0].show('most_active', fmat = 'xsd')