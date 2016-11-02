# -*- coding: utf-8 -*-
"""
Created on Wed Nov 02 14:29:52 2016

@author: mpnun
"""

import numpy as np
#from mpl_toolkits.mplot3d import Axes3D
from ase.neighborlist import NeighborList
from ase import Atoms
import random
from random import shuffle

import networkx as nx

from Helper import Helper

class Cat_structure_GT:
    
    def __init__(self):
        
        # Molecular objects
        self.bare_cat = Atoms()               # ASE molecular structure for non-functionalized catalyst
        self.site_cat = Atoms()               # ASE molecular structure with atoms representing binding sites
        self.functional_cat = Atoms()               # ASE molecular structure for functionalized catalyst
        
        # Functional groups
        self.functional_group_dict = {}         # Dictionary converting name of functional group to atoms object
        self.allowed_groups = {}                # Dictionary converting name of site type to list of functional groups allowed on that site        
        
        # Graph objects
        self.graph = nx.Graph()         # graph object representing the lattice
        self.graph_pos = {}             # keeps track of node locations for plotting

    def ASE_to_graph(self, element_to_sitetype, nn_dist = 1.25):        # I'm not sure how the cutoff distance works exactly...

        # Find neighbors based on distances
        rad_list = nn_dist / 2 * np.ones(self.site_cat.get_number_of_atoms())               # list of neighboradii for each site
        neighb_list = NeighborList(rad_list, self_interaction=False)
        neighb_list.build(self.site_cat)

        # Build graph nodes
        self.graph = nx.Graph()         # initialize as an empty graph
        self.graph_pos = {}             # initialize graph positions as an empty dictionary
        for i in range(self.site_cat.get_number_of_atoms()):
            self.graph.add_node(i, site_type = element_to_sitetype[self.site_cat.get_chemical_symbols()[i]], occupancy = 'empty')
            self.graph_pos[i] = self.site_cat.positions[i][[0,1]]       # keep only the x and y coordinates for plotting the graph
        
        # Add graph edges
        for i in range(self.site_cat.get_number_of_atoms()):
            for j in neighb_list.neighbors[i]:
                self.graph.add_edge(i, j)

    def show_graph(self):
        nx.draw(self.graph, self.graph_pos)
        
    def functionalize(self):

        self.functional_cat = self.bare_cat

        for site_ind in range(self.graph.number_of_nodes()):
            new_group = self.functional_group_dict[self.graph.node[site_ind]['occupancy']].copy()
            for atom_ind in range(new_group.get_number_of_atoms()):
                new_group.positions[atom_ind,:] = new_group.positions[atom_ind,:] + self.site_cat.positions[site_ind,:]
            self.functional_cat.extend(new_group)
            
    def randomize_occs(self):       # give each site a random occupancy which it is allowed to have

        for site_ind in range(self.graph.number_of_nodes()):
            allowed_groups = self.allowed_groups[ self.graph.node[site_ind]['site_type'] ]
            i = np.random.randint(low=0, high=len(allowed_groups))
            self.graph.node[site_ind]['occupancy'] = allowed_groups[i]
            
#    def seed_func_groups(self, n_ads, fg_name, site_type_name):            # Need to finish implementing this
#        print 'seeding ' + str(n_ads) + ' of ' + fg_name + ' on site type ' + site_type_name
#
#        sites_of_type = []
#        
#        for i in range(n_ads):
#            self.graph.node[sites_of_type[i]]['occupancy'] = fg_name