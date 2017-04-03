# Use this for converting Wei's model

import sys
import os
import numpy as np
import matplotlib as mat
import matplotlib.pyplot as plt
import copy
import random

sys.path.append('/home/vlachos/mpnunez/ase')
from ase.build import fcc111
from ase.io import read
from ase.visualize import view
from ase.io import write
from ase import Atoms

import networkx as nx
import networkx.algorithms.isomorphism as iso

sys.path.append('/home/vlachos/mpnunez/Github/Zacros-Wrapper/zacros_wrapper')
from Lattice import Lattice as lat
#import zacros_wrapper.Lattice as lat

from template import dyno_struc


class Wei_NH3_model(dyno_struc):
    
    '''
    Handles a dynamic lattice for Wei's NH3 decomposition model
    Data taken from W. Guo and D.G. Vlachos, Nat. Commun. 6, 8619 (2015).
    '''
    
    def __init__(self):
        
        '''
        Call superclass constructor
        '''
        
        super(Wei_NH3_model, self).__init__()
        
        self.dim1 = 8
        self.dim2 = 8

        
    def build_template(self):
    
        '''
        Build a 5-layer Pt(111) slab and transmute the top layer to Ni
        '''
    
        self.atoms_template = fcc111('Pt', size=(self.dim1, self.dim2, 5), vacuum=15.0)

        coords = self.atoms_template.get_positions()
        a_nums = self.atoms_template.get_atomic_numbers()
        chem_symbs = self.atoms_template.get_chemical_symbols()
        
        # Change top layer atoms to Ni
        top_layer_z = np.max(coords[:,2])
        for atom_ind in range(len(self.atoms_template)):
            if coords[atom_ind,2] > top_layer_z - 0.1:
                a_nums[atom_ind] = 28
                chem_symbs[atom_ind] = 'Ni'
                
        self.atoms_template.set_atomic_numbers(a_nums)
        self.atoms_template.set_chemical_symbols(chem_symbs)
        
    
    def generate_defected(self):    
        
        '''
        Take occupancies and use them to build a defected structure
        '''
        
        self.atoms_defected = copy.deepcopy(self.atoms_template)
        
        n_var = self.dim1 * self.dim2
        n_fixed = 4 * self.dim1 * self.dim2
        n_tot = n_var + n_fixed
        
        delete_these = [False for i in range(n_tot)]
        for i in range(n_fixed, n_tot):
            if random.uniform(0, 1) < 0.5:
                delete_these[i] = True
                
        delete_these = np.array(delete_these)
        
        del self.atoms_defected[delete_these]
        
        self.occupancies = delete_these
        
    
    def template_to_KMC_lattice(self):
    
        '''
        Convert defected atoms object to a KMC lattice object
        '''
    
        # Build lattice for Ni atoms only
        Ni_lattice = lat()
        Ni_lattice.lattice_matrix = self.atoms_template.get_cell()[0:2, 0:2]
        Ni_lattice.site_type_names = ['Ni_occ', 'Ni_vac']
        Ni_lattice.site_type_inds = [1 for i in range(self.dim1 * self.dim2)]         # Ni atoms for top layer only
        
        # Change missing atoms to vacant sites
        for i in range(len(self.occupancies[ 4 * self.dim1 * self.dim2 : 5 * self.dim1 * self.dim2 : ])):
            if self.occupancies[4 * self.dim1 * self.dim2 + i]:
                Ni_lattice.site_type_inds[i] = 2
        
        # Assign coordinates to sites and find neighbors
        Ni_lattice.set_cart_coords( self.atoms_template.get_positions()[4 * self.dim1 * self.dim2 : 5 * self.dim1 * self.dim2 : , 0:2] )    # x and y coordinates of Ni atoms
        Ni_lattice.Build_neighbor_list()
        
        # Set up object KMC lattice
        self.KMC_lat = lat()
        self.KMC_lat.lattice_matrix = self.atoms_template.get_cell()[0:2, 0:2]
        self.KMC_lat.site_type_names = ['Ni_top', 'Ni_hollow', 'Ni_edge']
        
        # Build networkx graph to help identify sites
        Ni_graph = dyno_struc.KMC_lattice_to_graph(Ni_lattice)
        
        # Add top sites
        for i in range(len(Ni_graph)):
            if Ni_graph.node[i]['type'] == 'Ni_occ':
                self.KMC_lat.site_type_inds.append(1)
                self.KMC_lat.cart_coords.append( self.atoms_template.get_positions()[4 * self.dim1 * self.dim2 + i, 0:2:] )
        
        # Add hollow sites
        Ni_trimer = nx.Graph() 
        Ni_trimer.add_nodes_from(['A','B','C'])
        Ni_trimer.add_edges_from([['A','B'], ['B','C'], ['A','C']])
        Ni_trimer.node['A']['type'] = 'Ni_occ'
        Ni_trimer.node['B']['type'] = 'Ni_occ'
        Ni_trimer.node['C']['type'] = 'Ni_occ'
        
        GM = iso.GraphMatcher(Ni_graph, Ni_trimer, node_match=iso.categorical_node_match('type', 'Ni_occ'))
        for subgraph in GM.subgraph_isomorphisms_iter():
            inv_map = {v: k for k, v in subgraph.items()}
            A_ind = inv_map['A']
            B_ind = inv_map['B']
            C_ind = inv_map['C']
            if A_ind < B_ind and B_ind < C_ind:
                self.KMC_lat.site_type_inds.append(2)
                self.KMC_lat.cart_coords.append( np.mean( self.atoms_template.get_positions()[ [4 * self.dim1 * self.dim2 + A_ind, 4 * self.dim1 * self.dim2 + B_ind, 4 * self.dim1 * self.dim2 + C_ind] , 0:2:], axis=0) )
        
        ## Add edge sites
        Ni_edge = nx.Graph() 
        Ni_edge.add_nodes_from(['A','B','C'])
        Ni_edge.add_edges_from([['A','B'], ['B','C'], ['A','C']])
        Ni_edge.node['A']['type'] = 'Ni_occ'
        Ni_edge.node['B']['type'] = 'Ni_occ'
        Ni_edge.node['C']['type'] = 'Ni_vac'
        GM2 = iso.GraphMatcher(Ni_graph, Ni_edge, node_match=iso.categorical_node_match('type', 'Ni_occ'))
        for subgraph in GM2.subgraph_isomorphisms_iter():
            inv_map = {v: k for k, v in subgraph.items()}
            A_ind = inv_map['A']
            B_ind = inv_map['B']
            C_ind = inv_map['C']
            if A_ind < B_ind:
                self.KMC_lat.site_type_inds.append(3)
                self.KMC_lat.cart_coords.append( np.mean( self.atoms_template.get_positions()[ [4 * self.dim1 * self.dim2 + A_ind, 4 * self.dim1 * self.dim2 + B_ind, 4 * self.dim1 * self.dim2 + C_ind] , 0:2:], axis=0) )
        
        self.KMC_lat.set_cart_coords(np.array(self.KMC_lat.cart_coords))        # Take list of coordinates and make it a vector
        
        
        
        
if __name__ == "__main__":

    '''
    Check to see that our lattice is being built correctly
    '''

    x = Wei_NH3_model()
    x.build_template()
    x.generate_defected()
    x.template_to_KMC_lattice()
    
    plt = x.KMC_lat.PlotLattice()
    plt.savefig(os.path.join(x.path, 'kmc_lattice.png'))
    plt.close()
    write('defected.xyz', x.atoms_defected, format = 'xyz')
