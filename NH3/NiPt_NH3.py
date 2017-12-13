import numpy as np
import copy

from ase.neighborlist import NeighborList

import networkx as nx
import networkx.algorithms.isomorphism as iso

from dynamic_cat import dynamic_cat
from zacros_wrapper.Lattice import Lattice as lat


class NiPt_NH3(dynamic_cat):
    
    '''
    Handles a dynamic lattice for Wei's NH3 decomposition model
    Data taken from W. Guo and D.G. Vlachos, Nat. Commun. 6, 8619 (2015).
    '''
    
    def __init__(self, dimen = 16):
        
        '''
        Modify the template atoms object and build defected graph
        '''
        
        dynamic_cat.__init__(self, dim1 = dimen, dim2 = dimen, fixed_layers = 4, variable_layers = 1)       # Call parent class constructor # 
        
        self.KMC_lat = None                 # Zacros Wrapper lattice object 
        
        
        '''
        Transmute the top layer to Ni
        '''  

        coords = self.atoms_template.get_positions()
        a_nums = self.atoms_template.get_atomic_numbers()
        chem_symbs = self.atoms_template.get_chemical_symbols()
        
        # Change top layer atoms to Cu and 2nd to top layer to Ni
        for atom_ind in range(3*self.atoms_per_layer,4*self.atoms_per_layer):
            a_nums[atom_ind] = 28
            chem_symbs[atom_ind] = 'Ni'
        for atom_ind in range(4*self.atoms_per_layer,5*self.atoms_per_layer):
            a_nums[atom_ind] = 29
            chem_symbs[atom_ind] = 'Cu'
                
        self.atoms_template.set_atomic_numbers(a_nums)
        self.atoms_template.set_chemical_symbols(chem_symbs)
        
        self.variable_atoms = range(3*self.atoms_per_layer,4*self.atoms_per_layer)  # 2nd layer from the top is variable
        self.variable_occs = [1 for i in self.variable_atoms]
        
        '''
        Build graph
        '''
        
        n_at = len(self.atoms_template)
        
        self.defected_graph = nx.Graph()
        self.defected_graph.add_nodes_from(range(n_at))       # nodes indexed with integers
        
        for i in range(n_at):       # Assign elements and site types in graph
        
            if self.atoms_template.get_chemical_symbols()[i] == 'Cu':
                self.defected_graph.node[i]['element'] = 'vacuum'
            else:
                self.defected_graph.node[i]['element'] = self.atoms_template.get_chemical_symbols()[i]
                
            self.defected_graph.node[i]['site_type'] = None
        
        # Find neighbors based on distances
        self.atoms_template.set_pbc([True, True, False])
        rad_list = ( 2.77 + 0.2) / 2 * np.ones(n_at)               # list of neighbor radii for each site
        neighb_list = NeighborList(rad_list, self_interaction = False)      # set bothways = True to include both ways
        neighb_list.build(self.atoms_template)
        
        # Add edges between adjacent atoms
        for i in range(n_at):
            for j in neighb_list.neighbors[i]:
                self.defected_graph.add_edge(i, j)
                
        self.occs_to_atoms()
        self.occs_to_graph()
        
        
    def occs_to_graph(self):
        '''
        Build graph from occupancies
        Assign elements in the graph according to occupancies
        '''
        
        for i in range(len(self.variable_occs)):
        
            if self.variable_occs[i]:
                self.defected_graph.node[self.variable_atoms[i]]['element'] = 'Ni'
            else:
                self.defected_graph.node[self.variable_atoms[i]]['element'] = 'vacancy'


    def graph_to_occs(self):
        '''
        Convert graph representation of defected structure to occupancies
        '''
        
        for i in range(len(self.variable_occs)):
        
            if self.defected_graph.node[self.variable_atoms[i]]['element'] == 'Ni':
                self.variable_occs[i] = 1
            elif self.defected_graph.node[self.variable_atoms[i]]['element'] == 'vacancy':
                self.variable_occs[i] = 0
            else:
                raise NameError('Invalid element type in graph for variable atom.')
                
    
    def get_site_data(self):
        '''
        Evaluate the contribution to the current from each site
        
        :returns: Array site currents for each active site, 3 columns for 3 sites per atom (top, fcc hollow, hcp hollow)
        NEED TO IMPLEMENT THIS
        ''' 
        site_data = np.zeros([self.atoms_per_layer,3])
        
        for i in range(len(self.variable_occs)):
 
            neighb_dict = self.defected_graph[self.variable_atoms[i]]
            n_Ni_neighbs = 0
            n_vac_neighbs = 0
            for key in neighb_dict:
                if self.defected_graph.node[key]['element'] == 'Ni':
                    n_Ni_neighbs += 1
                elif self.defected_graph.node[key]['element'] == 'vacancy':
                    n_vac_neighbs += 1
            
            if self.defected_graph.node[self.variable_atoms[i]]['element'] == 'Ni':     # Only active if a Ni atom is present
                site_data[i,0] = n_Ni_neighbs * n_vac_neighbs
        
        return site_data
        
    
    def flip_atom(self, ind):
        
        '''
        ind: Index of the site to change.
        If it is a Ni, change it to a vacancy.
        If it is a vacancy, change it to Ni.
        '''  
        super(NiPt_NH3, self).flip_atom(ind)     # Call super class method to change the occupancy vector
        
        isos_removed = []
        element_from = self.defected_graph.node[ind]['element'] 
        
        if element_from == 'vacancy':
            
            self.defected_graph.node[ind]['element'] = 'Ni'
            self.atoms_missing[ind] = False
            
        elif element_from == 'Ni':
            
            self.defected_graph.node[ind]['element'] = 'vacancy'
            self.atoms_missing[ind] = True
                
        else:
            print element_from
            raise NameError('Flipped atoms must be Ni or a vacancy.')
            
        # Remove isomorphisms which have site ind as the old property
        for isom in self.target_isos:
            if ind in isom:
                if x.target_graph.node[ isom[ind] ]['element'] == element_from:     # This should always be true because element matching was imposed in the isomorphism
                    isos_removed.append(isom)
                    
        for isom in isos_removed:
            self.target_isos.remove(isom)

        # Add new isomorphisms in a neighborhood around site ind
        neighb = nx.ego_graph(self.defected_graph, ind, radius = self.target_diam)         # Take radius around new node
        GM4 = iso.GraphMatcher(neighb, x.target_graph, node_match=iso.categorical_node_match('element','Au') )
        for subgraph in GM4.subgraph_isomorphisms_iter():
            if ind in subgraph:
                self.target_isos.append(subgraph)        
        
                    
    def graph_to_KMClattice(self):
    
        '''
        Converts defected graph to a KMC lattice object
        '''
        
        if self.defected_graph is None:
            raise NameError('Lattice graph not yet defined.')
        
        n_at = len(self.atoms_template)        
        
        for i in range(n_at):           # Set all site types as unknown
            self.defected_graph.node[i]['site_type'] = None
        
        # Prepare a tetrahedron graph which will be useful
        tet_graph = nx.Graph() 
        tet_graph.add_nodes_from(['A', 'B', 'C', 'D'])
        tet_graph.add_edges_from([['A','B'], ['B','C'], ['C','A'], ['A', 'D'], ['B', 'D'], ['C', 'D']])
        
        '''
        Add site types one by one
        '''
        
        # 4. Ni corner
        for i in range(n_at):
            if self.defected_graph.node[i]['element'] == 'Ni':   # Ni atoms not defined as sites  
                self.defected_graph.node[i]['site_type'] = 4
        
        
        # 3. Ni_top
        for i in range(n_at):
            if self.defected_graph.node[i]['element'] == 'Ni':
                
                # count the number of neighbors that are Ni edge sites
                n_Ni_neighbs = 0                 
                for neighb in self.defected_graph.neighbors(i):               # look though the neighbors of the Ni fcc sites (1)
                    if self.defected_graph.node[neighb]['element'] == 'Ni':
                        n_Ni_neighbs += 1
                        
                if n_Ni_neighbs == 6:
                    self.defected_graph.node[i]['site_type'] = 3
        
        
        
        # 5. Ni edge
        # Has two adjacent vacancies as neighbors. The other 4 neighbors must be Ni.
        mini_graph = nx.Graph() 
        mini_graph.add_nodes_from(['A', 'B', 'C'])
        mini_graph.add_edges_from([['A','B'], ['B','C'], ['C','A']])
        mini_graph.node['A']['element'] = 'Ni'
        mini_graph.node['B']['element'] = 'vacancy'
        mini_graph.node['C']['element'] = 'vacancy'
        GM = iso.GraphMatcher(self.defected_graph, mini_graph, node_match=iso.categorical_node_match('element', 'Ni'))
        for subgraph in GM.subgraph_isomorphisms_iter():
            inv_map = {v: k for k, v in subgraph.items()}
            A_ind = inv_map['A']
            
            # count the number of neighbors that are Ni edge sites
            n_Ni_neighbs = 0                 
            for neighb in self.defected_graph.neighbors(A_ind):               # look though the neighbors of the Ni atom (1)
                if self.defected_graph.node[neighb]['element'] == 'Ni':
                    n_Ni_neighbs += 1
                    
            if n_Ni_neighbs == 4:
                self.defected_graph.node[A_ind]['site_type'] = 5
        
        
        
        
        # 1. Ni_fcc
        mini_graph = copy.deepcopy(tet_graph)
        mini_graph.node['A']['element'] = 'Ni'
        mini_graph.node['B']['element'] = 'Ni'
        mini_graph.node['C']['element'] = 'Ni'
        mini_graph.node['D']['element'] = 'vacuum'
        GM = iso.GraphMatcher(self.defected_graph, mini_graph, node_match=iso.categorical_node_match('element', 'Ni'))
        for subgraph in GM.subgraph_isomorphisms_iter():
            inv_map = {v: k for k, v in subgraph.items()}
            D_ind = inv_map['D']
            self.defected_graph.node[D_ind]['site_type'] = 1
        
        # 2. Ni_hcp
        mini_graph = copy.deepcopy(tet_graph)
        mini_graph.node['A']['element'] = 'Ni'
        mini_graph.node['B']['element'] = 'Ni'
        mini_graph.node['C']['element'] = 'Ni'
        mini_graph.node['D']['element'] = 'Pt'
        GM = iso.GraphMatcher(self.defected_graph, mini_graph, node_match=iso.categorical_node_match('element', 'Ni'))
        for subgraph in GM.subgraph_isomorphisms_iter():
            inv_map = {v: k for k, v in subgraph.items()}
            D_ind = inv_map['D']
            self.defected_graph.node[D_ind]['site_type'] = 2
        
        

        
        # 6. Pt_fcc
        for i in range(n_at):
            if self.defected_graph.node[i]['element'] == 'vacancy':
                self.defected_graph.node[i]['site_type'] = 6
        
        # 8. Pt_top
        mini_graph = copy.deepcopy(tet_graph)
        mini_graph.node['A']['element'] = 'vacancy'
        mini_graph.node['B']['element'] = 'vacancy'
        mini_graph.node['C']['element'] = 'vacancy'
        mini_graph.node['D']['element'] = 'Pt'
        GM = iso.GraphMatcher(self.defected_graph, mini_graph, node_match=iso.categorical_node_match('element', 'Ni'))
        for subgraph in GM.subgraph_isomorphisms_iter():
            inv_map = {v: k for k, v in subgraph.items()}
            D_ind = inv_map['D']
            self.defected_graph.node[D_ind]['site_type'] = 8
        
        # 7. Pt_hcp 
        mini_graph = copy.deepcopy(tet_graph)
        mini_graph.node['A']['site_type'] = 8
        mini_graph.node['B']['site_type'] = 8
        mini_graph.node['C']['site_type'] = 8
        mini_graph.node['D']['site_type'] = None
        GM = iso.GraphMatcher(self.defected_graph, mini_graph, node_match=iso.categorical_node_match('site_type', 8))
        for subgraph in GM.subgraph_isomorphisms_iter():
            inv_map = {v: k for k, v in subgraph.items()}
            D_ind = inv_map['D']
            self.defected_graph.node[D_ind]['site_type'] = 7
        
        
        
        # 12. h4
        mini_graph = copy.deepcopy(tet_graph)
        mini_graph.node['A']['element'] = 'Ni'
        mini_graph.node['B']['element'] = 'vacancy'
        mini_graph.node['C']['element'] = 'vacancy'
        mini_graph.node['D']['element'] = 'vacuum'
        GM = iso.GraphMatcher(self.defected_graph, mini_graph, node_match=iso.categorical_node_match('element', 'Ni'))
        for subgraph in GM.subgraph_isomorphisms_iter():
            inv_map = {v: k for k, v in subgraph.items()}
            B_ind = inv_map['B']
            C_ind = inv_map['C']
            D_ind = inv_map['D']
            self.defected_graph.node[B_ind]['site_type'] = 10
            self.defected_graph.node[C_ind]['site_type'] = 10
            self.defected_graph.node[D_ind]['site_type'] = 12
        
        
        
        # 14. s2 and 11. f4
        mini_graph = copy.deepcopy(tet_graph)
        mini_graph.node['A']['element'] = 'Ni'
        mini_graph.node['B']['element'] = 'Ni'
        mini_graph.node['C']['element'] = 'vacancy'
        mini_graph.node['D']['element'] = 'vacuum'
        GM = iso.GraphMatcher(self.defected_graph, mini_graph, node_match=iso.categorical_node_match('element', 'Ni'))
        for subgraph in GM.subgraph_isomorphisms_iter():
            inv_map = {v: k for k, v in subgraph.items()}
            C_ind = inv_map['C']
            D_ind = inv_map['D']
            self.defected_graph.node[C_ind]['site_type'] = 11
            self.defected_graph.node[D_ind]['site_type'] = 14
            
        # 15. s1 and 10. f3
        mini_graph = copy.deepcopy(tet_graph)
        mini_graph.node['A']['element'] = 'Ni'
        mini_graph.node['B']['element'] = 'Ni'
        mini_graph.node['C']['element'] = 'vacancy'
        mini_graph.node['D']['element'] = 'Pt'
        GM = iso.GraphMatcher(self.defected_graph, mini_graph, node_match=iso.categorical_node_match('element', 'Ni'))
        for subgraph in GM.subgraph_isomorphisms_iter():
            inv_map = {v: k for k, v in subgraph.items()}
            D_ind = inv_map['D']
            self.defected_graph.node[D_ind]['site_type'] = 15
        
        # 16. f1 and 17. f2
        for i in range(n_at):
            if self.defected_graph.node[i]['site_type'] == 1:     # look though the neighbors of the Ni fcc sites (1)
                n_edges = 0                 # count the number of neighbors that are Ni edge sites
                for neighb in self.defected_graph.neighbors(i):
                    if self.defected_graph.node[neighb]['site_type'] == 5 or self.defected_graph.node[neighb]['site_type'] == 4:
                        n_edges += 1
                if n_edges == 1:
                    self.defected_graph.node[i]['site_type'] = 17
                elif n_edges >= 2:
                    self.defected_graph.node[i]['site_type'] = 16
        
        
        # 18. h1
        mini_graph = copy.deepcopy(tet_graph)
        mini_graph.node['A']['site_type'] = 3
        mini_graph.node['B']['site_type'] = 3
        mini_graph.node['C']['site_type'] = 5
        mini_graph.node['D']['site_type'] = 2
        GM = iso.GraphMatcher(self.defected_graph, mini_graph, node_match=iso.categorical_node_match('site_type', 8))
        for subgraph in GM.subgraph_isomorphisms_iter():
            inv_map = {v: k for k, v in subgraph.items()}
            D_ind = inv_map['D']
            self.defected_graph.node[D_ind]['site_type'] = 18
        
        # 19. h2
        mini_graph = copy.deepcopy(tet_graph)
        mini_graph.node['A']['site_type'] = 3
        mini_graph.node['B']['site_type'] = 5
        mini_graph.node['C']['site_type'] = 5
        mini_graph.node['D']['site_type'] = 2
        GM = iso.GraphMatcher(self.defected_graph, mini_graph, node_match=iso.categorical_node_match('site_type', 8))
        for subgraph in GM.subgraph_isomorphisms_iter():
            inv_map = {v: k for k, v in subgraph.items()}
            D_ind = inv_map['D']
            self.defected_graph.node[D_ind]['site_type'] = 19
        
        
        # 9. h5
        for i in range(n_at):
            if self.defected_graph.node[i]['site_type'] == 6:
                i_pos = self.atoms_template.get_positions()[i,0:2:]
                s1_in_range = False
                
                for j in range(n_at):
                    if self.defected_graph.node[j]['site_type'] == 15:
                        j_pos = self.atoms_template.get_positions()[j,0:2:]

                        if np.linalg.norm( i_pos - j_pos ) < 2.77 * 2:
                            s1_in_range = True
                
                if s1_in_range:
                    self.defected_graph.node[i]['site_type'] = 9

        
        # 13. h6
        mini_graph = copy.deepcopy(tet_graph)
        mini_graph.node['A']['site_type'] = 8
        mini_graph.node['B']['site_type'] = 8
        mini_graph.node['C']['site_type'] = None
        mini_graph.node['D']['site_type'] = None
        GM = iso.GraphMatcher(self.defected_graph, mini_graph, node_match=iso.categorical_node_match('site_type', 8))
        for subgraph in GM.subgraph_isomorphisms_iter():
            inv_map = {v: k for k, v in subgraph.items()}
            A_ind = inv_map['A']
            D_ind = inv_map['D']
            if self.atoms_template.get_positions()[D_ind,2] - self.atoms_template.get_positions()[A_ind,2] < -0.1: # lower layer than top Pt
                self.defected_graph.node[D_ind]['site_type'] = 13
            
            
        '''
        Build KMC lattice
        '''
        
        # Set up object KMC lattice
        self.KMC_lat = lat()
        self.KMC_lat.text_only = False
        self.KMC_lat.lattice_matrix = self.atoms_template.get_cell()[0:2, 0:2]
        
        # Wei Nature site names
        #self.KMC_lat.site_type_names = ['Ni_fcc', 'Ni_hcp', 'Ni_top', 'Ni_corner', 'Ni_edge', 'Pt_fcc', 'Pt_hcp', 
        #    'Pt_top', 'h5', 'f3', 'f4', 'h4', 'h6', 's2', 's1', 'f1', 'f2', 'h1', 'h2']
            
        # Older site names
        #self.KMC_lat.site_type_names = ['fcc_Ni',	'hcp_Ni',	'top_Ni',	'top_corner_Ni',	'top_edge_Ni',	'fcc_Pt',
        #'hcp_Pt',	'top_Pt', 'hcp_2edge_Pt_3fcc',	'fcc_edge_Pt_3fcc',	'fcc_edge_Pt_3hcp',	'hcp_edge_Pt_3fcc',	'hcp_edge_Pt_3hcp',
        #    'step_100',	'step_110',	'fcc_edge_Ni_3fcc',	'fcc_edge_Ni_3hcp',	'hcp_edge_Ni_3fcc',	'hcp_edge_Ni_3hcp']
        
        self.KMC_lat.site_type_names = ['fcc_Ni',	'top_Ni', 'top_edge_Ni',
            'fcc_edge_Ni_3fcc',	'fcc_edge_Ni_3hcp',	'hcp_edge_Ni_3fcc',	'hcp_edge_Ni_3hcp']
        
        # All atoms with a defined site type
        site_ind_dict = {1:1, 3:2, 5:3, 16:4, 17:5, 18:6, 19:7}
        cart_coords_list = []
        for i in range(n_at):
            if self.defected_graph.node[i]['site_type'] in site_ind_dict:
                self.KMC_lat.site_type_inds.append( site_ind_dict[ self.defected_graph.node[i]['site_type'] ])
                cart_coords_list.append( self.atoms_template.get_positions()[i, 0:2:] )

        self.KMC_lat.set_cart_coords(cart_coords_list)
        self.KMC_lat.Build_neighbor_list(cut = 2.77 + 0.1)