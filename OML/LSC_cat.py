import numpy as np
import copy

from ase.neighborlist import NeighborList

import networkx as nx
import networkx.algorithms.isomorphism as iso

from dynamic_cat import dynamic_cat
from zacros_wrapper.Lattice import Lattice as lat
import time

class LSC_cat(dynamic_cat):
    
    '''
    Catalyst with a very simple KMC lattice and linear site coupling (LSC)
    '''
    
    def __init__(self, dimen = 12):
        
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
        

    def graph_to_KMClattice_V2(self):
        '''
        Trying to make a faster implementation, not used yet
        '''
        all_syms = self.generate_all_translations_and_rotations()
        all_trans = all_syms[0:-1:3,:]
        
        n_at = len(self.atoms_template)
        d = self.dim1
        site_types = [None for i in range(n_at)]
        
        for i in range(len(self.variable_occs)):
            st_ind = i+3*self.atoms_per_layer
            hep = all_trans[i,[0, 2*d-1, d, 1, d*(d-1)+1, d*(d-1), d-1]]
            if np.sum(hep) == 7:
                site_types[st_ind] = 1      # terrace
            elif np.sum(hep) == 5:              # might be an edge
                site_types[st_ind] = 1      # terrace
            # otherwise, it is neigher a terrace nor edge site
            
        
        
        '''
        Build KMC lattice
        '''
        
        # Set up object KMC lattice
        self.KMC_lat = lat()
        self.KMC_lat.text_only = False
        self.KMC_lat.lattice_matrix = self.atoms_template.get_cell()[0:2, 0:2]
        
        
        self.KMC_lat.site_type_names = ['top_Ni', 'top_edge_Ni']
        
        
        # All atoms with a defined site type
        cart_coords_list = []
        self.var_ind_kmc_sites = []
        for i in range(n_at):
            if not site_types[i] is None:
                self.KMC_lat.site_type_inds.append(site_types[i])
                cart_coords_list.append( self.atoms_template.get_positions()[i, 0:2:] )
                self.var_ind_kmc_sites.append(i-3*self.atoms_per_layer)
        t11 = time.time() 
        if len(self.KMC_lat.site_type_inds) > 0:
            self.KMC_lat.set_cart_coords(cart_coords_list)
            self.KMC_lat.Build_neighbor_list(cut = 2.77 + 0.1)
        t12 = time.time() 
        print 'Neighbor time: ' + str(t12 -t11)
    
    def graph_to_KMClattice(self):
    
        '''
        Converts defected graph to a KMC lattice object
        Should speed this up - It is limiting optimization with the analytical model
        '''
        
        if self.defected_graph is None:
            raise NameError('Lattice graph not yet defined.')
        
        n_at = len(self.atoms_template)        
        
        for i in range(n_at):           # Set all site types as unknown
            self.defected_graph.node[i]['site_type'] = None
        
        
        '''
        Add site types one by one
        '''
        
        
        # 3. Ni_top
        for i in range(n_at):
            if self.defected_graph.node[i]['element'] == 'Ni':
                
                # count the number of neighbors that are Ni edge sites
                n_Ni_neighbs = 0                 
                for neighb in self.defected_graph.neighbors(i):               # look though the neighbors of the Ni fcc sites (1)
                    if self.defected_graph.node[neighb]['element'] == 'Ni':
                        n_Ni_neighbs += 1
                        
                if n_Ni_neighbs == 6:
                    self.defected_graph.node[i]['site_type'] = 1
        
        
        
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
            
            # count the number of neighbors that are Ni atoms
            n_Ni_neighbs = 0                 
            for neighb in self.defected_graph.neighbors(A_ind):               # look though the neighbors of the Ni atom (1)
                if self.defected_graph.node[neighb]['element'] == 'Ni':
                    n_Ni_neighbs += 1
                    
            if n_Ni_neighbs == 4:
                self.defected_graph.node[A_ind]['site_type'] = 2
        
            
        '''
        Build KMC lattice
        '''

        # Set up object KMC lattice
        self.KMC_lat = lat()
        self.KMC_lat.text_only = False
        self.KMC_lat.lattice_matrix = self.atoms_template.get_cell()[0:2, 0:2]
        
        self.KMC_lat.site_type_names = ['top_Ni', 'top_edge_Ni']
        
        # All atoms with a defined site type
        cart_coords_list = []
        self.var_ind_kmc_sites = []
        for i in range(n_at):
            if not self.defected_graph.node[i]['site_type'] is None:
                self.KMC_lat.site_type_inds.append(self.defected_graph.node[i]['site_type'])
                cart_coords_list.append( self.atoms_template.get_positions()[i, 0:2:] )
                self.var_ind_kmc_sites.append(i-3*self.atoms_per_layer)
        
        if len(self.KMC_lat.site_type_inds) > 0:
            self.KMC_lat.set_cart_coords(cart_coords_list)
            self.KMC_lat.Build_neighbor_list(cut = 2.77 + 0.1)
       
    
    def eval_struc_rate_anal(self,x):
        '''
        Evaluate the structure rate analytically
        :param x: Occupancies
        :returns: Total structure rate (not normalized)
        '''
        self.assign_occs(x)
        self.graph_to_KMClattice()
        site_rates_anal = self.compute_site_rates_lsc()
        return np.sum(site_rates_anal)
        
    
    def compute_site_rates_lsc(self):
        '''
        Compute analytical solution to the site rates for a certain catalyst
        :param kmc_lat: KMC lattice
        :returns: site rates
        should make this use sparse matrices
        '''
        
        kmc_lat = self.KMC_lat
        
        # Rate constants
        k_ads_terrace = 1.0
        k_des_terrace = 1.0
        k_diff = 1.0
        k_des_edge = 1.0
        
        # Build matrices
        
        kmc_lat.site_type_inds
        kmc_lat.neighbor_list
        n_sites = len(kmc_lat.site_type_inds)
        
        site_rates = np.zeros(len(self.variable_occs))
        
        # If there are no sites
        if n_sites == 0:
            return site_rates
        
        A = np.zeros([n_sites,n_sites])
        b = np.zeros(n_sites)
        A_rate = np.zeros([n_sites,n_sites])
        
        for site_ind in range(n_sites):
            
            # Adsorption/desorption
            if kmc_lat.site_type_inds[site_ind] == 1:
                A[site_ind,site_ind] += -1 * (k_ads_terrace + k_des_terrace)
                b[site_ind] += -k_ads_terrace
            elif kmc_lat.site_type_inds[site_ind] == 2:
                A[site_ind,site_ind] += -1 * k_des_edge
                A_rate[site_ind,site_ind] = k_des_edge
            
            # Diffusion
            for site_ind_2 in range(n_sites):
            
                if [site_ind,site_ind_2] in kmc_lat.neighbor_list or [site_ind_2,site_ind] in kmc_lat.neighbor_list:
                    A[site_ind,site_ind] += -1 * k_diff
                    A[site_ind,site_ind_2] += k_diff
        
        # Solve linear system of equations for the coverages
        theta = np.linalg.solve(A, b)
        kmc_site_rates = A_rate.dot(theta)
        
        
        for i in range(len(kmc_site_rates)):
            site_rates[self.var_ind_kmc_sites[i]] = kmc_site_rates[i]
        
        return site_rates