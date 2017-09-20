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

class cat_structure():
    
    '''
    Catalyst structure with defects
    '''
    
    def __init__(self, met_name = None, facet = None, dim1 = None, dim2 = None):
        
        self.dim1 = dim1
        self.dim2 = dim2
        
        self.metal = None
        self.atoms_obj_template = None
        self.active_atoms = None
        self.variable_atoms = None
        self.variable_occs = None
        self.template_graph = None
        self.defected_graph = None
        self.surface_area = None                 # surface area of the slab in square angstroms
        
        self.evaluated = False
        self.current_density = None
        self.surf_eng = None
        self.active_CN = 9                      # CN must be less than or equal to this to be active
        
        if not met_name is None:
        
            '''
            Build the slab
            '''        
            
            self.metal = metal(met_name)
            
            if facet == '111' or facet == 111:
                self.atoms_obj_template = fcc111(met_name, size=(dim1, dim2, 4), vacuum=15.0)
                self.active_CN = 9
            elif facet == '100' or facet == 100:
                self.atoms_obj_template = fcc100(met_name, size=(dim1, dim2, 4), vacuum=15.0)
                self.active_CN = 8
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
        
        :param coverage: coverage of the random structure. If None, coverage
            will be chosen from the uniform distribution between 0 and 1
        '''
        
        x_rand = [0 for i in range(len(self.variable_atoms)) ]
        
        if coverage is None:
            coverage = random.random()
        n_occupancies = int( round( coverage * len(self.variable_atoms) ) )
        occupied_sites = random.sample(range(len(x_rand)), n_occupancies)
        for i in occupied_sites:
            x_rand[i] = 1
        return x_rand
        
    
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
        
        self.compute_variable_occs()
        
    
    def get_Nnn(self):
        '''
        For each active atom, print the number of nearest neighbors that are also active
        '''
        atom_graph = self.defected_graph
        for i in self.active_atoms:
            if atom_graph.is_node(i):
                if atom_graph.get_coordination_number(i) <= self.active_CN:
                    
                    gcn = atom_graph.get_generalized_coordination_number(i, 12)
                    
                    Nnn = 0
                    for j in atom_graph.get_neighbors(i):
                        if j in self.active_atoms:
                            if atom_graph.is_node(j):
                                if atom_graph.get_coordination_number(j) <= self.active_CN:
                                    Nnn += 1
                    
                    print [gcn, Nnn]
        
    
    def get_site_currents(self):
        '''
        Evaluate the contribution to the current from each site
        
        :returns: List of site currents for each active site
        '''

        curr_list = [0. for i in range(len(self.active_atoms))]
        for i in range(len(self.active_atoms)):
            site_ind = self.active_atoms[i]
            if self.defected_graph.is_node(site_ind):
                if self.defected_graph.get_coordination_number(site_ind) <= self.active_CN:
                    gcn = self.defected_graph.get_generalized_coordination_number(site_ind, 12)
                    BE_OH = self.metal.get_OH_BE(gcn)
                    BE_OOH = self.metal.get_OOH_BE(gcn)
                    curr_list[i] = ORR_rate(BE_OH, BE_OOH)
                    
        return curr_list
                    
                    
    def eval_current_density(self, normalize = True):
        
        '''
        :param normalize: current density [mA/cm^2]
        
        Not normalized: current [mA]
        '''       
        
        site_currents = self.get_site_currents()
        J = sum(site_currents)
        
        if normalize:
            J = J / ( self.surface_area * 1.0e-16)          # normalize by surface area (in square centimeters)
  
        return J
        
    
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
                #E_form += self.metal.E_coh * ( 1 - atom_graph.get_coordination_number(i) / 12.0 )
                
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
    

    def show(self, x = None, fname = 'structure_1', fmat = 'png'):
                
        '''
        Print image of surface
        '''        
        
        # Build defected graph
        if not x is None:
            self.defected_graph = self.template_graph.copy_data()
            for i in range(len(x)):
                if x[i] == 0:
                    self.defected_graph.remove_vertex(self.variable_atoms[i])
        
        # Build ASE atoms object from defected graph
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
        
        if fmat == 'png':
            write(fname + '.png', defect_atoms_obj )
        elif fmat == 'xsd':
            defect_atoms_obj.set_pbc(True)
            write(fname + '.xsd', defect_atoms_obj, format = fmat )
        elif fmat == 'povray':
            write(fname + '.pov', defect_atoms_obj )

    
    def compute_variable_occs(self):
        self.variable_occs = [0 for i in range(len(self.variable_atoms))]
        ind = 0
        for i in self.variable_atoms:
            if self.defected_graph.is_node(i):
                self.variable_occs[ind] = 1
            ind += 1
    
    
    def generate_all_translations(self):
        '''
        Generate a symmetery matrix which has all possible translations of the
        sites within the unit cell
        '''
        
        all_translations = []
        n_var = len(self.variable_atoms)
        for var_ind in range(n_var):
            d1, d2 = self.var_ind_to_sym_inds(var_ind)    
            all_translations.append( self.variable_shift( d1, d2) )
                
        return np.array(all_translations)

        
    def variable_shift(self, shift1, shift2):
        '''
        Permute occupancies according to symmetry
        
        :param shift1: Number of indices to translate along first axis
        
        :param shift2: Number of indices to translate along second axis
        '''
        
        n_var = len(self.variable_atoms)
        new_occs = np.zeros(n_var)
        
        for var_ind in range(n_var):
            d1, d2 = self.var_ind_to_sym_inds(var_ind)
            map_from_ind = self.sym_inds_to_var_ind( d1 - shift1 , d2 - shift2 )
            new_occs[var_ind] = self.variable_occs[ map_from_ind ]
            
        return new_occs
        
    
    def sym_inds_to_var_ind(self, sym_ind1, sym_ind2):
        sym_ind1 = sym_ind1 % self.dim1
        sym_ind2 = sym_ind2 % self.dim2
        return sym_ind2 * self.dim1 + sym_ind1
    
    def var_ind_to_sym_inds(self,var_ind):
        return var_ind % self.dim1, var_ind / self.dim1