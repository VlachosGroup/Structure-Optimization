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
from ase.neighborlist import NeighborList

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
    
    Pt_Pt_1nn_dist = 2.77       # angstrom
    
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
    
        # Set periodic boundary conditions - periodic in x and y directions, aperiodic in the z direction
        self.atoms_template.set_pbc([True, True, False])
        self.atoms_defected.set_pbc([True, True, False])
        n_at = len(self.atoms_template)
        template_cc = self.atoms_template.get_positions()
        Ni_z = np.max(template_cc[:,2])
        bot_lay_z = np.min(template_cc[:,2])
        Pt_fcc_z = bot_lay_z + 0.25 * (Ni_z - bot_lay_z)
        Pt_hcp_z = bot_lay_z + 0.50 * (Ni_z - bot_lay_z)
        Pt_top_z = bot_lay_z + 0.75 * (Ni_z - bot_lay_z)
        
        # Find neighbors based on distances
        rad_list = ( Wei_NH3_model.Pt_Pt_1nn_dist + 0.2) / 2 * np.ones(n_at)               # list of neighboradii for each site
        neighb_list = NeighborList(rad_list, self_interaction = False)      # set bothways = True to include both ways
        neighb_list.build(self.atoms_template)
        
        # Build graph
        mol_graph = nx.Graph()
        mol_graph.add_nodes_from(range(n_at))       # nodes indexed with integers
        pos_dict = {}
        
        for i in range(n_at):
            if self.occupancies[i]:
                if self.atoms_template.get_chemical_symbols()[i] == 'Ni':
                    mol_graph.node[i]['element'] = 'vacancy'
                elif self.atoms_template.get_chemical_symbols()[i] == 'Cu':
                    mol_graph.node[i]['element'] = 'vacuum'
                else:
                    print 'No element found'
            else:
                mol_graph.node[i]['element'] = self.atoms_template.get_chemical_symbols()[i]
            mol_graph.node[i]['site_type'] = None
            pos_dict[i] = self.atoms_template.get_positions()[i,0:2:]
        
        # Add edges between adjacent atoms
        for i in range(n_at):
            for j in neighb_list.neighbors[i]:
                mol_graph.add_edge(i, j)
        
        # Draw template graph
        #nx.draw(mol_graph, pos = pos_dict)    
        #plt.savefig('mol_graph.png')
        #plt.close()
        
        # Set up object KMC lattice
        self.KMC_lat = lat()
        self.KMC_lat.workingdir = self.path
        self.KMC_lat.lattice_matrix = self.atoms_template.get_cell()[0:2, 0:2]
        self.KMC_lat.site_type_names = ['Ni_fcc', 'Ni_hcp', 'Ni_top', 'Ni corner', 'Ni edge', 'Pt_fcc', 'Pt_hcp', 
            'Pt_top', 'h5', 'f3', 'f4', 'h4', 'h6', 's2', 's1', 'f1', 'f2', 'h1', 'h2']
        
        # Prepare a tetrahedron graph which will be useful
        tet_graph = nx.Graph() 
        tet_graph.add_nodes_from(['A', 'B', 'C', 'D'])
        tet_graph.add_edges_from([['A','B'], ['B','C'], ['C','A'], ['A', 'D'], ['B', 'D'], ['C', 'D']])
        
        '''
        Add site types one by one
        '''
        
        # 1. Ni_fcc
        mini_graph = copy.deepcopy(tet_graph)
        mini_graph.node['A']['element'] = 'Ni'
        mini_graph.node['B']['element'] = 'Ni'
        mini_graph.node['C']['element'] = 'Ni'
        mini_graph.node['D']['element'] = 'vacuum'
        GM = iso.GraphMatcher(mol_graph, mini_graph, node_match=iso.categorical_node_match('element', 'Ni'))
        for subgraph in GM.subgraph_isomorphisms_iter():
            inv_map = {v: k for k, v in subgraph.items()}
            D_ind = inv_map['D']
            mol_graph.node[D_ind]['site_type'] = 1
        
        # 2. Ni_hcp
        mini_graph = copy.deepcopy(tet_graph)
        mini_graph.node['A']['element'] = 'Ni'
        mini_graph.node['B']['element'] = 'Ni'
        mini_graph.node['C']['element'] = 'Ni'
        mini_graph.node['D']['element'] = 'Pt'
        GM = iso.GraphMatcher(mol_graph, mini_graph, node_match=iso.categorical_node_match('element', 'Ni'))
        for subgraph in GM.subgraph_isomorphisms_iter():
            inv_map = {v: k for k, v in subgraph.items()}
            D_ind = inv_map['D']
            mol_graph.node[D_ind]['site_type'] = 2
        
        # 3. Ni_top
        for i in range(n_at):
            if mol_graph.node[i]['element'] == 'Ni':
                mol_graph.node[i]['site_type'] = 3
        
        
        
        # 5. Ni edge
        mini_graph = nx.Graph() 
        mini_graph.add_nodes_from(['A', 'B', 'C'])
        mini_graph.add_edges_from([['A','B'], ['B','C'], ['C','A']])
        mini_graph.node['A']['element'] = 'Ni'
        mini_graph.node['B']['element'] = 'Ni'
        mini_graph.node['C']['element'] = 'vacancy'
        GM = iso.GraphMatcher(mol_graph, mini_graph, node_match=iso.categorical_node_match('element', 'Ni'))
        for subgraph in GM.subgraph_isomorphisms_iter():
            inv_map = {v: k for k, v in subgraph.items()}
            A_ind = inv_map['A']
            B_ind = inv_map['B']
            mol_graph.node[A_ind]['site_type'] = 5
            mol_graph.node[B_ind]['site_type'] = 5
        
        # 4. Ni corner
        for i in range(n_at):
            if mol_graph.node[i]['site_type'] == 5 or mol_graph.node[i]['site_type'] == 3:     # look though the neighbors of the Ni fcc sites (1)
                n_Ni_neighbs = 0                 # count the number of neighbors that are Ni edge sites
                for neighb in mol_graph.neighbors(i):
                    if mol_graph.node[neighb]['site_type'] == 3 or mol_graph.node[neighb]['site_type'] == 4 or mol_graph.node[neighb]['site_type'] == 5:
                        n_Ni_neighbs += 1
                if n_Ni_neighbs <= 3:
                    mol_graph.node[i]['site_type'] == 4

        
        # 6. Pt_fcc
        for i in range(n_at):
            if mol_graph.node[i]['element'] == 'vacancy':
                mol_graph.node[i]['site_type'] = 6
        
        # 8. Pt_top
        mini_graph = copy.deepcopy(tet_graph)
        mini_graph.node['A']['element'] = 'vacancy'
        mini_graph.node['B']['element'] = 'vacancy'
        mini_graph.node['C']['element'] = 'vacancy'
        mini_graph.node['D']['element'] = 'Pt'
        GM = iso.GraphMatcher(mol_graph, mini_graph, node_match=iso.categorical_node_match('element', 'Ni'))
        for subgraph in GM.subgraph_isomorphisms_iter():
            inv_map = {v: k for k, v in subgraph.items()}
            D_ind = inv_map['D']
            mol_graph.node[D_ind]['site_type'] = 8
        
        # 7. Pt_hcp 
        mini_graph = copy.deepcopy(tet_graph)
        mini_graph.node['A']['site_type'] = 8
        mini_graph.node['B']['site_type'] = 8
        mini_graph.node['C']['site_type'] = 8
        mini_graph.node['D']['site_type'] = None
        GM = iso.GraphMatcher(mol_graph, mini_graph, node_match=iso.categorical_node_match('site_type', 8))
        for subgraph in GM.subgraph_isomorphisms_iter():
            inv_map = {v: k for k, v in subgraph.items()}
            D_ind = inv_map['D']
            mol_graph.node[D_ind]['site_type'] = 7
        
        # 9. h5
        
        
        # 12. h4
        
        # 13. h6
        
        # 14. s2 and 11. f4
        mini_graph = copy.deepcopy(tet_graph)
        mini_graph.node['A']['element'] = 'Ni'
        mini_graph.node['B']['element'] = 'Ni'
        mini_graph.node['C']['element'] = 'vacancy'
        mini_graph.node['D']['element'] = 'vacuum'
        GM = iso.GraphMatcher(mol_graph, mini_graph, node_match=iso.categorical_node_match('element', 'Ni'))
        for subgraph in GM.subgraph_isomorphisms_iter():
            inv_map = {v: k for k, v in subgraph.items()}
            A_ind = inv_map['A']
            B_ind = inv_map['B']
            C_ind = inv_map['C']
            D_ind = inv_map['D']
            mol_graph.node[A_ind]['site_type'] = 5
            mol_graph.node[B_ind]['site_type'] = 5
            mol_graph.node[C_ind]['site_type'] = 11
            mol_graph.node[D_ind]['site_type'] = 14
            
        # 15. s1 and 10. f3
        mini_graph = copy.deepcopy(tet_graph)
        mini_graph.node['A']['element'] = 'Ni'
        mini_graph.node['B']['element'] = 'Ni'
        mini_graph.node['C']['element'] = 'vacancy'
        mini_graph.node['D']['element'] = 'Pt'
        GM = iso.GraphMatcher(mol_graph, mini_graph, node_match=iso.categorical_node_match('element', 'Ni'))
        for subgraph in GM.subgraph_isomorphisms_iter():
            inv_map = {v: k for k, v in subgraph.items()}
            A_ind = inv_map['A']
            B_ind = inv_map['B']
            D_ind = inv_map['D']
            mol_graph.node[A_ind]['site_type'] = 5
            mol_graph.node[B_ind]['site_type'] = 5
            mol_graph.node[D_ind]['site_type'] = 15
        
        # 16. f1 and 17. f2
        for i in range(n_at):
            if mol_graph.node[i]['site_type'] == 1:     # look though the neighbors of the Ni fcc sites (1)
                n_edges = 0                 # count the number of neighbors that are Ni edge sites
                for neighb in mol_graph.neighbors(i):
                    if mol_graph.node[neighb]['site_type'] == 5 or mol_graph.node[neighb]['site_type'] == 4:
                        n_edges += 1
                if n_edges == 1:
                    mol_graph.node[i]['site_type'] = 17
                elif n_edges >= 2:
                    mol_graph.node[i]['site_type'] = 16
        
        
        # 18. h1
        mini_graph = copy.deepcopy(tet_graph)
        mini_graph.node['A']['site_type'] = 3
        mini_graph.node['B']['site_type'] = 3
        mini_graph.node['C']['site_type'] = 5
        mini_graph.node['D']['site_type'] = None
        GM = iso.GraphMatcher(mol_graph, mini_graph, node_match=iso.categorical_node_match('site_type', 8))
        for subgraph in GM.subgraph_isomorphisms_iter():
            inv_map = {v: k for k, v in subgraph.items()}
            D_ind = inv_map['D']
            mol_graph.node[D_ind]['site_type'] = 18
        
        # 19. h2
        mini_graph = copy.deepcopy(tet_graph)
        mini_graph.node['A']['site_type'] = 3
        mini_graph.node['B']['site_type'] = 5
        mini_graph.node['C']['site_type'] = 5
        mini_graph.node['D']['site_type'] = None
        GM = iso.GraphMatcher(mol_graph, mini_graph, node_match=iso.categorical_node_match('site_type', 8))
        for subgraph in GM.subgraph_isomorphisms_iter():
            inv_map = {v: k for k, v in subgraph.items()}
            D_ind = inv_map['D']
            mol_graph.node[D_ind]['site_type'] = 19
        
        
        '''
        Finish building KMC lattice
        '''
        
        # All atoms with a defined site type
        cart_coords_list = []
        for i in range(n_at):
            if not mol_graph.node[i]['site_type'] is None:
                self.KMC_lat.site_type_inds.append(mol_graph.node[i]['site_type'])
                cart_coords_list.append( self.atoms_template.get_positions()[i, 0:2:] )
        
        self.KMC_lat.set_cart_coords(cart_coords_list)
        
        #self.KMC_lat.Build_neighbor_list(cut = Wei_NH3_model.Pt_Pt_1nn_dist + 0.2)
        
        
if __name__ == "__main__":

    '''
    Check to see that our lattice is being built correctly
    '''
    
    
    x = Wei_NH3_model()
    #x.Load_defect( 'NiPt_template.xsd', 'NiPt_defected.xsd' )
    x.Load_defect( 'new_template.xsd', 'new_defected.xsd' )
    x.template_to_KMC_lattice()
    
    x.KMC_lat.Write_lattice_input()     # write lattice_input.dat
    
    # Create a png file with the lattice drawn
    plt = x.KMC_lat.PlotLattice()
    plt.savefig(os.path.join(x.path, 'kmc_lattice.png'))
    plt.close()
    
    # Draw a graph of Wei's lattice
    #y = lat()
    #y.Read_lattice_output('Wei_NH3_lattice_output.txt')
    #plt = y.PlotLattice()
    #plt.savefig('Wei.png')
    #plt.close()
