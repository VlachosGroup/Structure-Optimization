# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 12:42:16 2016

@author: mpnun
"""

import numpy as np
#from mpl_toolkits.mplot3d import Axes3D
from ase.neighborlist import NeighborList
from ase import Atoms
import random
from random import shuffle

from Helper import Helper

class Pattern:
    
    def __init__(self, ads_types = [0], neighbs = []):
        self.ads_types = ads_types
        self.adj_mat = np.zeros([len(ads_types), len(ads_types)])
        
        for i in range(len(neighbs)):
            self.adj_mat[ neighbs[i][0]-1 , neighbs[i][1]-1 ] = 1
            self.adj_mat[ neighbs[i][1]-1 , neighbs[i][0]-1 ] = 1

class Cat_structure:
    
    def __init__(self):
        
        # Molecular objects
        self.bare_cat = ''               # ASE molecular structure for non-functionalized catalyst
        self.functional_cat = ''               # ASE molecular structure for non-functionalized catalyst
        self.functional_group_list = []         # List of atoms objects of functional groups
        
        # Lattice occupancies
        self.site_locs = []             # Cartesian coordinates for occupancy sites
        self.site_occs = []             # list of which functional groups are present at each site
        self.site_types = []
        self.func_groups_allowed = []       # list of functional groups allowed to occupy each site type
        self.adj_mat = []
        self.E = 0

    def randomize_occs(self):           # give each site a random occupancy which it is allowed to have
        
        self.site_occs = [0 for i in range(self.site_locs.shape[0])]

        for occ_ind in range(len(self.site_occs)):
            allowed_groups = self.func_groups_allowed[self.site_types[occ_ind]]
            i = np.random.randint(low=0, high=len(allowed_groups))
            self.site_occs[occ_ind] = allowed_groups[i]
            
    def seed_functional_groups(self, site_type, n_each_func_group):
        
        new_occs = []
        for i in range(len(n_each_func_group)):
            for j in range(n_each_func_group[i]):
                new_occs.append(i)
        shuffle(new_occs)

        ind = 0
        for site_ind in range(len(self.site_occs)):
            if self.site_types[site_ind] == site_type:
                self.site_occs[site_ind] = new_occs[ind]
                ind += 1

    def functionalize(self):            # Need to figure out how to plot in 3-D

        self.functional_cat = self.bare_cat

        for site_ind in range(len(self.site_occs)):
            if self.site_occs[site_ind] > 0:
                new_group = self.functional_group_list[self.site_occs[site_ind]-1].copy()
                for atom_ind in range(new_group.positions.shape[0]):
                    new_group.positions[atom_ind,:] = new_group.positions[atom_ind,:] + self.site_locs[site_ind,:]
                self.functional_cat.extend(new_group)

#        self.functional_cat = Helper.sort_atoms(self.functional_cat)           # sorts atoms by element
    
    def find_neighbs(self, nn_dist = 1.6):
        
        rad_list = nn_dist / 2 * np.ones(len(self.site_occs))
        neighb_list = NeighborList(rad_list, self_interaction=False)
        
        fake_mol = Atoms('Li' + str(len(self.site_occs)), self.site_locs)   
        fake_mol.cell = self.bare_cat.cell
        
        neighb_list.build(fake_mol)
        
        adj_mat = np.zeros((fake_mol.get_number_of_atoms(), fake_mol.get_number_of_atoms()))
        
        for i in range(fake_mol.get_number_of_atoms()):
            for j in neighb_list.neighbors[i]:
                adj_mat[i,j] = 1
                adj_mat[j,i] = 1
                
        self.adj_mat = adj_mat
        
    def count_pairs(self, fg1, fg2):
        occ1 = np.zeros(len(self.site_occs))
        occ2 = np.zeros(len(self.site_occs))
        
        for i in range(len(self.site_occs)):
            if self.site_occs[i] == fg1:
                occ1[i] = 1
            if self.site_occs[i] == fg2:
                occ2[i] = 1
                
        return np.dot(np.dot(occ1, self.adj_mat), np.transpose(occ2))
    
    def count_pattern(self, pat):
        return 0
    
    def evaluate_eng(self):
        self.E = 0
        
    def mutate(self):       # Change the functional group at a site
        rand_site_ind = np.random.randint(low=0, high=len(self.site_occs))
        allowed_groups = self.func_groups_allowed[self.site_types[rand_site_ind]]
        self.site_occs[rand_site_ind] = random.choice(allowed_groups)
        
    def mutate_CE(self):    # Exchange functional groups between adjacent sites of the same type
        rand_site_ind = np.random.randint(low=0, high=len(self.site_occs))
        allowed_groups = self.func_groups_allowed[self.site_types[rand_site_ind]]
        self.site_occs[rand_site_ind] = random.choice(allowed_groups)