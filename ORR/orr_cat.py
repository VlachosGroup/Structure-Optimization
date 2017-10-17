import numpy as np
import copy

from ase.neighborlist import NeighborList

from metal import metal
from ORR import ORR_rate
from graph_theory import Graph
from cat_optimization.dynamic_cat import dynamic_cat

class orr_cat(dynamic_cat):
    
    '''
    Oxygen reduction reaction catalyst structure with defects
    '''
    
    def __init__(self, met_name = 'Pt', facet = '111', dim1 = 12, dim2 = 12):
        
        dynamic_cat.__init__(self, met_name = met_name, facet = facet, dim1 = dim1, dim2 = dim1, fixed_layers = 3, variable_layers = 1)       # Call parent class constructor
        
        self.metal = None
        
        self.template_graph = None
        self.defected_graph = None
        self.active_atoms = None            # Atoms which contribute to the current density
        
        if facet == '111':
            self.active_CN = 9                      # CN must be less than or equal to this to be active    
        elif facet == '100':
            self.active_CN = 8
            
        self.active_atoms = range(2 * self.atoms_per_layer, 4 * self.atoms_per_layer)
        self.metal = metal(met_name)
        
        '''
        Build template graph
        '''
        
        # Find neighbors based on distances
        rad_list = ( 2.77 + 0.2 ) / 2 * np.ones(len(self.atoms_template))               # list of neighboradii for each site
        neighb_list = NeighborList(rad_list, self_interaction = False)      # set bothways = True to include both ways
        neighb_list.build(self.atoms_template)
        
        self.template_graph = Graph()
        for i in range(len(self.atoms_template)):
            self.template_graph.add_vertex(i)
        
        for i in range(len(self.atoms_template)):
            for j in neighb_list.neighbors[i]:
                self.template_graph.add_edge([i,j])
                
        self.defected_graph = copy.deepcopy(self.template_graph)
        
        self.occs_to_atoms()
        self.occs_to_graph()
    
    
    def graph_to_occs(self):
        '''
        Convert graph representation of defected structure to occupancies
        '''
        self.variable_occs = [0 for i in range(len(self.variable_atoms))]
        ind = 0
        for i in self.variable_atoms:
            if self.defected_graph.is_node(i):
                self.variable_occs[ind] = 1
            ind += 1
            
            
    def occs_to_graph(self, x = None):
        '''
        Build graph from occupancies
        '''
        if x is None:
            x = self.variable_occs
        else:
            self.variable_occs = x
        
        self.defected_graph = self.template_graph.copy_data()
        for i in range(len(x)):
            if x[i] == 0:
                self.defected_graph.remove_vertex(self.variable_atoms[i])
        
    
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
        
    
    def get_site_data(self):
        '''
        Evaluate the contribution to the current from each site
        
        :returns: Array site currents for each active site
        '''

        curr_list = [0. for i in range(len(self.active_atoms))]
        for i in range(len(self.active_atoms)):
            site_ind = self.active_atoms[i]
            if self.defected_graph.is_node(site_ind):
                if self.defected_graph.get_coordination_number(site_ind) <= self.active_CN:
                    gcn = self.defected_graph.get_generalized_coordination_number(site_ind, 12)
                    BE_OH = self.metal.get_OH_BE(gcn)
                    BE_OOH = self.metal.get_OOH_BE(gcn)
                    #curr_list[i] = ORR_rate(BE_OH, BE_OOH)
                    curr_list[i] = gcn
                    
        curr_list = np.transpose( np.array(curr_list).reshape([2,self.atoms_per_layer]) )  
        
        #return np.sum(curr_list, axis = 1)
        return curr_list[:,0]
                    
    def eval_current_density(self, normalize = True, site_currents = None):
        
        '''
        :param normalize: current density [mA/cm^2]
        
        Not normalized: current [mA]
        '''       
        
        if site_currents is None:
            site_currents = self.get_site_data()
        J = np.sum(site_currents)
        
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
        
    
    def flip_atom(self, ind):
        
        '''
        If atom number ind is present in the defected graph, remove it.
        If it is not present, add it and all edges to adjacent atoms.
        '''        
        super(orr_cat, self).flip_atom(ind)     # Call super class method to change the occupancy vector
        
        if self.defected_graph.is_node(ind):
            self.defected_graph.remove_vertex(ind)
        else:
            self.defected_graph.add_vertex(ind)
            for neighb in self.template_graph.get_neighbors(ind):
                if self.defected_graph.is_node(neighb):
                    self.defected_graph.add_edge([ind, neighb])

            
    def eval_x(self, y):
    
        y = np.array(y)
        self.build_graph(x = y)
        return self.get_OFs()
        
    
    def get_OF(self):
        
        '''
        :param weights: Weights for the objective functions
        :returns: A single objective function that is a linear superposition of the other objectives
        '''

        return self.weights[0] * self.eval_surface_energy() + self.weights[1] * self.eval_current_density()
        
    
    def get_OFs(self):
        
        '''
        Evaluate the objective functions
        :returns: 2-ple of surface energy and current density
        '''
        
        return self.eval_surface_energy(), self.eval_current_density()
        
        
    def show(self, fname = 'structure_1', fmat = 'png', transmute_top = True, chop_top = False):
        '''
        Use super class method with top layer transmuted to display
        '''
        super(orr_cat, self).show(fname = fname, fmat = fmat, transmute_top = transmute_top)