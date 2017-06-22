# Use this for converting Wei's model

import sys
import os
import numpy as np
import matplotlib as mat
mat.use('Agg')
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
        
        self.dim1 = None
        self.dim2 = None

        # Need a class variable that lists the variable nodes of the graph
        
    def build_template(self, d1, d2):
    
        '''
        Build a 5-layer Pt(111) slab and transmute the top layer to Ni
        '''
    
        self.dim1 = d1
        self.dim2 = d2
    
        self.ASE_template = fcc111('Pt', size=(self.dim1, self.dim2, 5), vacuum=15.0)

        coords = self.ASE_template.get_positions()
        a_nums = self.ASE_template.get_atomic_numbers()
        chem_symbs = self.ASE_template.get_chemical_symbols()
        
        # Change top layer atoms to Ni
        top_layer_z = np.max(coords[:,2])
        for atom_ind in range(len(self.ASE_template)):
            if coords[atom_ind,2] > top_layer_z - 0.1:
                a_nums[atom_ind] = 28
                chem_symbs[atom_ind] = 'Ni'
                
        self.ASE_template.set_atomic_numbers(a_nums)
        self.ASE_template.set_chemical_symbols(chem_symbs)
        
    
    def randomize_atoms_missing(self):    
        
        '''
        Randomize the atoms_missing vector
        '''
        
        self.ASE_defected = copy.deepcopy(self.ASE_template)
        
        n_var = self.dim1 * self.dim2
        n_fixed = 4 * self.dim1 * self.dim2
        n_tot = n_var + n_fixed
        
        delete_these = [False for i in range(n_tot)]
        for i in range(n_fixed, n_tot):
            if random.uniform(0, 1) < 0.5:
                delete_these[i] = True
                
        delete_these = np.array(delete_these)
        
        del self.ASE_defected[delete_these]
        
        self.atoms_missing = delete_these
        
        
    def atoms_missing_to_graph3D(self):
    
        '''
        Converts defected ASE atoms object to a NetworkX graph object
        '''
    
        # Set periodic boundary conditions - periodic in x and y directions, aperiodic in the z direction
        self.ASE_template.set_pbc([True, True, False])
        self.ASE_defected.set_pbc([True, True, False])
        n_at = len(self.ASE_template)
        
        # Find neighbors based on distances
        rad_list = ( Wei_NH3_model.Pt_Pt_1nn_dist + 0.2) / 2 * np.ones(n_at)               # list of neighbor radii for each site
        neighb_list = NeighborList(rad_list, self_interaction = False)      # set bothways = True to include both ways
        neighb_list.build(self.ASE_template)
        
        # Build graph
        self.molecular_NetX = nx.Graph()
        self.molecular_NetX.add_nodes_from(range(n_at))       # nodes indexed with integers
        pos_dict = {}                                       # Used to visualize the graph if needed
        
        self.variable_atoms = []
        
        for i in range(n_at):
        
            if self.ASE_template.get_chemical_symbols()[i] == 'Ni':
                self.variable_atoms.append(i)
        
            if self.atoms_missing[i]:
                if self.ASE_template.get_chemical_symbols()[i] == 'Ni':
                    self.molecular_NetX.node[i]['element'] = 'vacancy'
                elif self.ASE_template.get_chemical_symbols()[i] == 'Cu':
                    self.molecular_NetX.node[i]['element'] = 'vacuum'
                else:
                    print 'No element found'
                    
            else:
                self.molecular_NetX.node[i]['element'] = self.ASE_template.get_chemical_symbols()[i]
                
            self.molecular_NetX.node[i]['site_type'] = None
            pos_dict[i] = self.ASE_template.get_positions()[i,0:2:]
        
        # Add edges between adjacent atoms
        for i in range(n_at):
            for j in neighb_list.neighbors[i]:
                self.molecular_NetX.add_edge(i, j)
        
    
    def flip_atom(self, ind):
        
        '''
        ind: Index of the site to change.
        If it is a Ni, change it to a vacancy.
        If it is a vacancy, change it to Ni.
        '''        
        
        if self.molecular_NetX.node[ind]['element'] == 'vacancy':
            self.molecular_NetX.node[ind]['element'] = 'Ni'
            self.atoms_missing[ind] = False
        elif self.molecular_NetX.node[ind]['element'] == 'Ni':
            self.molecular_NetX.node[ind]['element'] == 'vacancy'
            self.atoms_missing[ind] = True
        else:
            raise NameError('Flipped atoms must be Ni or a vacancy.')
                    
    
    def graph3D_to_KMC_lattice(self):
    
        '''
        Converts NetworkX graph object to a KMC lattice object
        '''
        
        if self.molecular_NetX is None:
            raise NameError('Lattice graph not yet defined.')
        
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
        GM = iso.GraphMatcher(self.molecular_NetX, mini_graph, node_match=iso.categorical_node_match('element', 'Ni'))
        for subgraph in GM.subgraph_isomorphisms_iter():
            inv_map = {v: k for k, v in subgraph.items()}
            D_ind = inv_map['D']
            self.molecular_NetX.node[D_ind]['site_type'] = 1
        
        # 2. Ni_hcp
        mini_graph = copy.deepcopy(tet_graph)
        mini_graph.node['A']['element'] = 'Ni'
        mini_graph.node['B']['element'] = 'Ni'
        mini_graph.node['C']['element'] = 'Ni'
        mini_graph.node['D']['element'] = 'Pt'
        GM = iso.GraphMatcher(self.molecular_NetX, mini_graph, node_match=iso.categorical_node_match('element', 'Ni'))
        for subgraph in GM.subgraph_isomorphisms_iter():
            inv_map = {v: k for k, v in subgraph.items()}
            D_ind = inv_map['D']
            self.molecular_NetX.node[D_ind]['site_type'] = 2
        
        # 3. Ni_top
        for i in range(n_at):
            if self.molecular_NetX.node[i]['element'] == 'Ni':
                self.molecular_NetX.node[i]['site_type'] = 3
        
        
        
        # 5. Ni edge
        mini_graph = nx.Graph() 
        mini_graph.add_nodes_from(['A', 'B', 'C'])
        mini_graph.add_edges_from([['A','B'], ['B','C'], ['C','A']])
        mini_graph.node['A']['element'] = 'Ni'
        mini_graph.node['B']['element'] = 'Ni'
        mini_graph.node['C']['element'] = 'vacancy'
        GM = iso.GraphMatcher(self.molecular_NetX, mini_graph, node_match=iso.categorical_node_match('element', 'Ni'))
        for subgraph in GM.subgraph_isomorphisms_iter():
            inv_map = {v: k for k, v in subgraph.items()}
            A_ind = inv_map['A']
            B_ind = inv_map['B']
            self.molecular_NetX.node[A_ind]['site_type'] = 5
            self.molecular_NetX.node[B_ind]['site_type'] = 5
        
        # 4. Ni corner
        for i in range(n_at):
            if self.molecular_NetX.node[i]['site_type'] == 5 or self.molecular_NetX.node[i]['site_type'] == 3:     
                n_Ni_neighbs = 0                 # count the number of neighbors that are Ni edge sites
                for neighb in self.molecular_NetX.neighbors(i):               # look though the neighbors of the Ni fcc sites (1)
                    if self.molecular_NetX.node[neighb]['site_type'] == 3 or self.molecular_NetX.node[neighb]['site_type'] == 4 or self.molecular_NetX.node[neighb]['site_type'] == 5:
                        n_Ni_neighbs += 1
                if n_Ni_neighbs <= 3:
                    self.molecular_NetX.node[i]['site_type'] = 4

        
        # 6. Pt_fcc
        for i in range(n_at):
            if self.molecular_NetX.node[i]['element'] == 'vacancy':
                self.molecular_NetX.node[i]['site_type'] = 6
        
        # 8. Pt_top
        mini_graph = copy.deepcopy(tet_graph)
        mini_graph.node['A']['element'] = 'vacancy'
        mini_graph.node['B']['element'] = 'vacancy'
        mini_graph.node['C']['element'] = 'vacancy'
        mini_graph.node['D']['element'] = 'Pt'
        GM = iso.GraphMatcher(self.molecular_NetX, mini_graph, node_match=iso.categorical_node_match('element', 'Ni'))
        for subgraph in GM.subgraph_isomorphisms_iter():
            inv_map = {v: k for k, v in subgraph.items()}
            D_ind = inv_map['D']
            self.molecular_NetX.node[D_ind]['site_type'] = 8
        
        # 7. Pt_hcp 
        mini_graph = copy.deepcopy(tet_graph)
        mini_graph.node['A']['site_type'] = 8
        mini_graph.node['B']['site_type'] = 8
        mini_graph.node['C']['site_type'] = 8
        mini_graph.node['D']['site_type'] = None
        GM = iso.GraphMatcher(self.molecular_NetX, mini_graph, node_match=iso.categorical_node_match('site_type', 8))
        for subgraph in GM.subgraph_isomorphisms_iter():
            inv_map = {v: k for k, v in subgraph.items()}
            D_ind = inv_map['D']
            self.molecular_NetX.node[D_ind]['site_type'] = 7
        
        
        
        # 12. h4
        mini_graph = copy.deepcopy(tet_graph)
        mini_graph.node['A']['element'] = 'Ni'
        mini_graph.node['B']['element'] = 'vacancy'
        mini_graph.node['C']['element'] = 'vacancy'
        mini_graph.node['D']['element'] = 'vacuum'
        GM = iso.GraphMatcher(self.molecular_NetX, mini_graph, node_match=iso.categorical_node_match('element', 'Ni'))
        for subgraph in GM.subgraph_isomorphisms_iter():
            inv_map = {v: k for k, v in subgraph.items()}
            B_ind = inv_map['B']
            C_ind = inv_map['C']
            D_ind = inv_map['D']
            self.molecular_NetX.node[B_ind]['site_type'] = 10
            self.molecular_NetX.node[C_ind]['site_type'] = 10
            self.molecular_NetX.node[D_ind]['site_type'] = 12
        
        
        
        # 14. s2 and 11. f4
        mini_graph = copy.deepcopy(tet_graph)
        mini_graph.node['A']['element'] = 'Ni'
        mini_graph.node['B']['element'] = 'Ni'
        mini_graph.node['C']['element'] = 'vacancy'
        mini_graph.node['D']['element'] = 'vacuum'
        GM = iso.GraphMatcher(self.molecular_NetX, mini_graph, node_match=iso.categorical_node_match('element', 'Ni'))
        for subgraph in GM.subgraph_isomorphisms_iter():
            inv_map = {v: k for k, v in subgraph.items()}
            C_ind = inv_map['C']
            D_ind = inv_map['D']
            self.molecular_NetX.node[C_ind]['site_type'] = 11
            self.molecular_NetX.node[D_ind]['site_type'] = 14
            
        # 15. s1 and 10. f3
        mini_graph = copy.deepcopy(tet_graph)
        mini_graph.node['A']['element'] = 'Ni'
        mini_graph.node['B']['element'] = 'Ni'
        mini_graph.node['C']['element'] = 'vacancy'
        mini_graph.node['D']['element'] = 'Pt'
        GM = iso.GraphMatcher(self.molecular_NetX, mini_graph, node_match=iso.categorical_node_match('element', 'Ni'))
        for subgraph in GM.subgraph_isomorphisms_iter():
            inv_map = {v: k for k, v in subgraph.items()}
            D_ind = inv_map['D']
            self.molecular_NetX.node[D_ind]['site_type'] = 15
        
        # 16. f1 and 17. f2
        for i in range(n_at):
            if self.molecular_NetX.node[i]['site_type'] == 1:     # look though the neighbors of the Ni fcc sites (1)
                n_edges = 0                 # count the number of neighbors that are Ni edge sites
                for neighb in self.molecular_NetX.neighbors(i):
                    if self.molecular_NetX.node[neighb]['site_type'] == 5 or self.molecular_NetX.node[neighb]['site_type'] == 4:
                        n_edges += 1
                if n_edges == 1:
                    self.molecular_NetX.node[i]['site_type'] = 17
                elif n_edges >= 2:
                    self.molecular_NetX.node[i]['site_type'] = 16
        
        
        # 18. h1
        mini_graph = copy.deepcopy(tet_graph)
        mini_graph.node['A']['site_type'] = 3
        mini_graph.node['B']['site_type'] = 3
        mini_graph.node['C']['site_type'] = 5
        mini_graph.node['D']['site_type'] = 2
        GM = iso.GraphMatcher(self.molecular_NetX, mini_graph, node_match=iso.categorical_node_match('site_type', 8))
        for subgraph in GM.subgraph_isomorphisms_iter():
            inv_map = {v: k for k, v in subgraph.items()}
            D_ind = inv_map['D']
            self.molecular_NetX.node[D_ind]['site_type'] = 18
        
        # 19. h2
        mini_graph = copy.deepcopy(tet_graph)
        mini_graph.node['A']['site_type'] = 3
        mini_graph.node['B']['site_type'] = 5
        mini_graph.node['C']['site_type'] = 5
        mini_graph.node['D']['site_type'] = 2
        GM = iso.GraphMatcher(self.molecular_NetX, mini_graph, node_match=iso.categorical_node_match('site_type', 8))
        for subgraph in GM.subgraph_isomorphisms_iter():
            inv_map = {v: k for k, v in subgraph.items()}
            D_ind = inv_map['D']
            self.molecular_NetX.node[D_ind]['site_type'] = 19
        
        
        # 9. h5
        for i in range(n_at):
            if self.molecular_NetX.node[i]['site_type'] == 6:
                i_pos = self.ASE_template.get_positions()[i,0:2:]
                s1_in_range = False
                
                for j in range(n_at):
                    if self.molecular_NetX.node[j]['site_type'] == 15:
                        j_pos = self.ASE_template.get_positions()[j,0:2:]

                        if np.linalg.norm( i_pos - j_pos ) < Wei_NH3_model.Pt_Pt_1nn_dist * 2:
                            s1_in_range = True
                
                if s1_in_range:
                    self.molecular_NetX.node[i]['site_type'] = 9

        
        # 13. h6
        mini_graph = copy.deepcopy(tet_graph)
        mini_graph.node['A']['site_type'] = 8
        mini_graph.node['B']['site_type'] = 8
        mini_graph.node['C']['site_type'] = None
        mini_graph.node['D']['site_type'] = None
        GM = iso.GraphMatcher(self.molecular_NetX, mini_graph, node_match=iso.categorical_node_match('site_type', 8))
        for subgraph in GM.subgraph_isomorphisms_iter():
            inv_map = {v: k for k, v in subgraph.items()}
            A_ind = inv_map['A']
            D_ind = inv_map['D']
            if self.ASE_template.get_positions()[D_ind,2] - self.ASE_template.get_positions()[A_ind,2] < -0.1: # lower layer than top Pt
                self.molecular_NetX.node[D_ind]['site_type'] = 13
            
            
        '''
        Build KMC lattice
        '''
        
        # Set up object KMC lattice
        self.KMC_lat = lat()
        self.KMC_lat.workingdir = self.path
        self.KMC_lat.lattice_matrix = self.ASE_template.get_cell()[0:2, 0:2]
        
        # Wei Nature site names
        #self.KMC_lat.site_type_names = ['Ni_fcc', 'Ni_hcp', 'Ni_top', 'Ni_corner', 'Ni_edge', 'Pt_fcc', 'Pt_hcp', 
        #    'Pt_top', 'h5', 'f3', 'f4', 'h4', 'h6', 's2', 's1', 'f1', 'f2', 'h1', 'h2']
            
        # Older site names
        self.KMC_lat.site_type_names = ['fcc_Ni',	'hcp_Ni',	'top_Ni',	'top_corner_Ni',	'top_edge_Ni',	'fcc_Pt',
        'hcp_Pt',	'top_Pt', 'hcp_2edge_Pt_3fcc',	'fcc_edge_Pt_3fcc',	'fcc_edge_Pt_3hcp',	'hcp_edge_Pt_3fcc',	'hcp_edge_Pt_3hcp',
            'step_100',	'step_110',	'fcc_edge_Ni_3fcc',	'fcc_edge_Ni_3hcp',	'hcp_edge_Ni_3fcc',	'hcp_edge_Ni_3hcp']
        
        # All atoms with a defined site type
        cart_coords_list = []
        for i in range(n_at):
            if not self.molecular_NetX.node[i]['site_type'] is None:
                self.KMC_lat.site_type_inds.append(self.molecular_NetX.node[i]['site_type'])
                cart_coords_list.append( self.ASE_template.get_positions()[i, 0:2:] )
        
        self.KMC_lat.set_cart_coords(cart_coords_list)
        
        self.KMC_lat.Build_neighbor_list(cut = Wei_NH3_model.Pt_Pt_1nn_dist + 0.1)
        
        
    def eval_current_density(self):
    
        '''
        Evaluate the objective function
        '''
    
        mini_graph = nx.Graph() 
        mini_graph.add_nodes_from(['A', 'B', 'C', 'D'])
        mini_graph.add_edges_from([['A','B'], ['B','C'], ['C','A'], ['B', 'D'], ['C', 'D']])
        mini_graph.node['A']['element'] = 'Ni'
        mini_graph.node['B']['element'] = 'Ni'
        mini_graph.node['C']['element'] = 'Ni'
        mini_graph.node['D']['element'] = 'vacancy'
    
        GM = iso.GraphMatcher(self.molecular_NetX, mini_graph, node_match=iso.categorical_node_match('element', 'Ni'))
        OF = 0
        for subgraph in GM.subgraph_isomorphisms_iter():
            OF += 1
            
        # Count symmetry of the subgraph
        GM_sub = iso.GraphMatcher(mini_graph, mini_graph, node_match=iso.categorical_node_match('type','Au'))            
        symmetry_count = 0
        for subgraph in GM_sub.subgraph_isomorphisms_iter():
            symmetry_count += 1
        
        return OF / symmetry_count
        
        
    def eval_surface_energy(self, atom_graph = None, normalize = False):
                
        return 0.0
        
        
    def get_GCE_ind(self):
    
        if self.variable_atoms is None:
            raise NameError('Variable atoms not specified.')
        
        return random.choice( self.variable_atoms )
        
    
    def get_CE_inds(self):
    
        mini_graph = nx.Graph()
        mini_graph.add_nodes_from(['A', 'B'])
        mini_graph.add_edge(['A','B'])
        mini_graph.node['A']['element'] = 'Ni'
        mini_graph.node['B']['element'] = 'vacancy'
    
        GM = iso.GraphMatcher(self.molecular_NetX, mini_graph, node_match=iso.categorical_node_match('element', 'Ni'))
        n_choices = 0
        for subgraph in GM.subgraph_isomorphisms_iter():
            n_choices += 1
        
        if n_choices == 0:
            raise NameError('No available moves.')
        
        ind_to_choose = random.randrange(n_choices)
        
        ind = 0
        for subgraph in GM.subgraph_isomorphisms_iter():
            if ind == ind_to_choose:
                inv_map = {v: k for k, v in subgraph.items()}
                A_ind = inv_map['A']
                B_ind = inv_map['B']
                return [A_ind, B_ind]
            
            ind += 1
        
        
        
if __name__ == "__main__":

    '''
    Check to see that our lattice is being built correctly
    '''
    
    sys.setrecursionlimit(1500)             # Needed for large number of atoms
    
    # Create object
    x = Wei_NH3_model()
    
    use_files = True
    
    if not use_files:
    
        # Create structure and randomize
        x.build_template(12, 12)
        x.randomize_atoms_missing()
    
    else:
    
        # Read structures from xsd files
        x.ASE_template = read(os.path.join('External', 'template_2.xsd') , format = 'xsd')
        x.ASE_defected = read(os.path.join('External', 'defected_2.xsd') , format = 'xsd')
        x.ASE_to_atoms_missing()
    
    # Build NetworkX graph of the defected structure
    x.atoms_missing_to_graph3D()
    
    # Optimize
    print 'Starting optimization'
    x.optimize(n_snaps = 10)
    x.atoms_missing_to_ASE()
    write( os.path.join('External', 'optimized.png') , x.ASE_defected )
    
    # Generate KMC lattice
    #x.graph3D_to_KMC_lattice()
    #x.KMC_lattice_to_graph2D()
    
    '''
    Write output
    '''
    
    #x.KMC_lat.Write_lattice_input('.')     # write lattice_input.dat
    #
    ## Draw lattice in a png file
    #plt = x.KMC_lat.PlotLattice()
    #plt.savefig(os.path.join(x.path, 'kmc_lattice.png'))
    #plt.close()
    
