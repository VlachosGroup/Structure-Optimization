'''
Has template class to be inherited from for specific models
'''

import networkx as nx
import networkx.algorithms.isomorphism as iso

import sys
sys.path.append('/home/vlachos/mpnunez/Github/Zacros-Wrapper')
import zacros_wrapper.Lattice as lat
import zacros_wrapper as zw


class dyno_struc(object):
    
    '''
    Handles a dynamic structure upon which graphs and KMC lattices can be built
    '''
    
    def __init__(self):
    
        '''
        Inherit from object class and declare up class variables
        '''
    
        super(dyno_struc, self).__init__()
        
        self.path = '.'        
        
        self.atoms_template = None            # will be an ASE atoms object when 
        
        self.occupancies = None               # occupancies of atoms in the ASE atoms object
        self.atoms_defected = None            # ASE atoms object, with some atoms removed or transmuted
        
        self.KMC_lat = None                   # KMC lattice object
        self.lat_graph = None                 # networkx graph object
        
        self.fingerprint_graphs = None        # networkx graph object, subgraphs to
        self.fingerprint_counts = None        # number of each fingerprint present
        
    
    @staticmethod    
    def KMC_lattice_to_graph(kmc_lat):
    
        '''
        Convert KMC lattice object to a networkx graph object
        '''
        
        nx_graph = nx.Graph()
        nx_graph.add_nodes_from(range(len(kmc_lat.site_type_inds)))
        nx_graph.add_edges_from(kmc_lat.neighbor_list)

        for i in range(len(kmc_lat.site_type_inds)):
            nx_graph.node[i]['type'] = kmc_lat.site_type_names[kmc_lat.site_type_inds[i]-1]
            
        return nx_graph
            
            
    #def graph_to_KMC_lattice(self, site_coords):
    #
    #    '''
    #    Convert networkx graph object to KMC lattice object
    #    '''
    #
    #    self.KMC_lat = lat()
        
        
    
    def count_fingerprints(self):
        
        '''
        Enumerate subgraph isomorphisms
        '''
        
        n_fingerprints = len(self.fingerprint_graphs)
        self.fingerprint_counts = [0 for j in range(n_fingerprints)]
        
        for i in range(n_fingerprints):
            
            # Count subgraphs
            GM = iso.GraphMatcher(self.lat_graph, self.fingerprint_graphs[i], node_match=iso.categorical_node_match('type','Au'))
            
            n_subgraph_isos = 0
            for subgraph in GM.subgraph_isomorphisms_iter():
                n_subgraph_isos += 1
            
            # Count symmetry of the subgraph
            GM_sub = iso.GraphMatcher(self.fingerprint_graphs[i], self.fingerprint_graphs[i], node_match=iso.categorical_node_match('type','Au'))            
            
            symmetry_count = 0
            for subgraph in GM_sub.subgraph_isomorphisms_iter():
                symmetry_count += 1
                
            self.fingerprint_counts[i] = n_subgraph_isos / symmetry_count