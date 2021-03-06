import random
import numpy as np
import copy

from ase.io import read
from ase.io import write
from ase.build import fcc111, fcc100

class dynamic_cat(object):
    
    '''
    Template for defected catalyst facets to be optimized. Handles (111) and (100) facets.
    '''
    
    def __init__(self, met_name = 'Pt', facet = '111', dim1 = 12, dim2 = 12, fixed_layers = 3,
                variable_layers = 1):
        
        '''
        Build the slab and initialize vacancies, etc.
        :param met_name:            Name of the metal, Pt or Au
        :param facet:               (111) or (100) miller index plane
        :param dim1:                Atoms in unit cell along first dimension
        :param dim2:                Atoms in unit cell along second dimension
        :param fixed_layers:        Number of bottom layers that are never defected
        :param variable_layers:     Number of top layers with atoms that can be defected
        '''
        
        super(dynamic_cat, self).__init__()             # Call parent class constructor
        
        self.dim1 = dim1                                # Atoms in unit cell along first dimension
        self.dim2 = dim2                                # Atoms in unit cell along second dimension
        self.atoms_per_layer = self.dim1 * self.dim2    # Number of atoms per layer
        
        self.atoms_template = None                      # ASE atoms object that is a template for defected structures
        self.atoms_defected = None                      # ASE atoms object that models the defected structure
        self.variable_atoms = None                      # Atoms in the ASE atoms object which can be missing or tansmuted in defected structures
        self.variable_occs = None                       # 0 if same as the template structure, 1 if different
        self.surface_area = None                        # surface area of the slab in square angstroms
        self.defected_graph = None                      # graph which helps compute things, implemented in subclasses
        self.atom_last_moved = None                     # Index of the variable atom last changed, used to revert moves in simulated annealing
        
        '''
        Build the template atoms object as a slab
        '''        
        
        total_layers = fixed_layers + variable_layers
        
        if facet == '111' or facet == 111:
            self.atoms_template = fcc111(met_name, size=(dim1, dim2, total_layers), a = 3.9239, vacuum=15.0)
        elif facet == '100' or facet == 100:
            self.atoms_template = fcc100(met_name, size=(dim1, dim2, total_layers), a = 3.9239, vacuum=15.0)
        else:
            raise ValueError(str(facet) + ' is not a valid facet.')
            
        self.atoms_template.set_pbc([True, True, False])
        
        self.variable_atoms = range(fixed_layers * dim1 * dim2, total_layers * dim1 * dim2)     # Indices of the top layer atoms
        self.variable_occs = [1 for i in self.variable_atoms]           # All atoms present    
        self.surface_area = np.linalg.norm( np.cross( self.atoms_template.get_cell()[0,:], self.atoms_template.get_cell()[1,:] ) )
        
        
    def randomize(self, coverage = None, build_structure = False):
        
        '''
        Randomize the occupancies in the top layer
        
        :param coverage: coverage of the random structure. If None, coverage
            will be chosen from the uniform distribution between 0 and 1
        '''
        
        self.variable_occs = [0 for i in range(len(self.variable_atoms)) ]
        
        if coverage is None:
            coverage = random.random()
        n_occupancies = int( round( coverage * len(self.variable_atoms) ) )
        occupied_sites = random.sample(range(len(self.variable_occs)), n_occupancies)
        for i in occupied_sites:
            self.variable_occs[i] = 1
            
        if build_structure:
            self.occs_to_atoms()
            self.occs_to_graph()
            
        return self.variable_occs
        
    
    def atoms_to_occs(self, d_cut = 0.1):
    
        '''
        Convert defected atoms object to variable atom occupancies
        
        :param d_cut: distance cutoff in angstroms
        '''

        self.variable_occs = [0 for i in range(len(self.variable_atoms))]
        
        for i in range(len(self.variable_atoms)):
        
            atom_ind = self.variable_atoms[i]
        
            # Get position and atomic number of the template atom we are trying to find
            cart_coords = self.atoms_template.get_positions()[atom_ind, :]
            atomic_num = self.atoms_template.get_atomic_numbers()[atom_ind]
            
            defect_ind = 0      # index of defect atom which might be a match
            dist = 1.0      # distance between atoms we are trying to match
            
            match_found = False
            
            while (not match_found) and defect_ind < len(self.atoms_defected):
            
                defect_coords = self.atoms_defected.get_positions()[defect_ind, :]
                defect_an = self.atoms_defected.get_atomic_numbers()[defect_ind]
                dist = np.linalg.norm( cart_coords - defect_coords )
                match_found = (dist < d_cut) #and (defect_an == atomic_num)         # Second condition checks whether the elements match
                defect_ind += 1
                
            if match_found:
                self.variable_occs[i] = 1
        

    def occs_to_atoms(self):
        
        '''
        Copy the template atoms object to the defected atoms object. Then, delete all 
        atoms which are missing in the occupancy vector
        '''
        
        self.atoms_defected = copy.deepcopy(self.atoms_template)
        atoms_missing = [False for atom in self.atoms_template]
        for i in range(len(self.variable_occs)):
            atoms_missing[self.variable_atoms[i]] = (self.variable_occs[i] == 0)
        del self.atoms_defected[atoms_missing]
    
    
    def graph_to_atoms(self):
        '''
        Convert graph to defected atoms object
        '''
        self.graph_to_occs()
        self.occs_to_atoms()
        
        
    def atoms_to_graphs(self):
        '''
        Convert atoms object to graph
        '''
        self.occs_to_graph()
        self.atoms_to_occs()
    
    
    def assign_occs(self,x):
        '''
        Assign variable atom occupancies
        :param x: The new occupancy vector
        '''
        self.variable_occs = x
        self.occs_to_graph()
        self.occs_to_atoms()
        
    
    def load_defects(self,fname):
        '''
        Get defect information from a file
        '''
        self.atoms_defected = read(fname)
        self.atoms_defected.set_cell( self.atoms_template.get_cell() ) # copy cell from template
        self.atoms_to_occs()
        self.occs_to_graph()
        
    
    def flip_atom(self, ind):
        
        '''
        If atom number ind is present in the defected graph, remove it.
        If it is not present, add it and all edges to adjacent atoms.
        '''
        
        ind = self.variable_atoms.index(ind)    # switch from atom index to index of the occupancy vector
        if self.variable_occs[ind] == 1:
            self.variable_occs[ind] = 0
        elif self.variable_occs[ind] == 0:
            self.variable_occs[ind] = 1
        else:
            raise NameError('Invalid occupancy.')
        
        
    def geo_crossover(self, x1, x2, pt1 = 1, pt2 = 1):
        '''
        Geometry-based crossover. Partions the catalyst surface into regions in a checkerboard style and performs crossover.
        
        :param x1: Mom
        :param x2: Dad
        :param pt1: By default, use 1-point crossover in first dimension
        :param pt2: By default, use 1-point crossover in second dimension
        
        :returns: Two offspring
        '''
        
        x_bounds = [random.random() for i in xrange(pt1)]
        y_bounds = [random.random() for i in xrange(pt2)]
        
        frac_coords = self.atoms_template.get_scaled_positions()
        
        for i in xrange(len(x1)):
            
            # Find whether it is an even or odd cell
            score = 0
            for bound in x_bounds:
                if frac_coords[self.variable_atoms[i],0] > bound:
                    score += 1
            for bound in y_bounds:
                if frac_coords[self.variable_atoms[i],1] > bound:
                    score += 1
            
            # Swap if it is in an even cell
            if score % 2 == 0:
                x1[i], x2[i] = x2[i], x1[i]
        
        return x1, x2
    

    def rand_move(self):
        ''' Randomly change an occupancy in variable_atoms '''
        self.atom_last_moved = random.choice(self.variable_atoms)
        self.flip_atom(self.atom_last_moved)
    
    def revert_last(self):
        ''' Revert last move for simulated annealing '''
        self.flip_atom(self.atom_last_moved)
        
    
    def generate_all_translations(self, local = False):
        '''
        Generate a symmetery matrix which has all possible translations of the
        sites within the unit cell
        '''
        
        all_translations = []
        n_var = len(self.variable_atoms)
        for var_ind in range(n_var):
            d1, d2 = self.var_ind_to_sym_inds(var_ind)
            all_translations.append( self.translate( d1, d2) )

        return np.array(all_translations)

        
    def translate(self, shift1, shift2, old_vec = None):
        '''
        Permute occupancies according to symmetry
        
        :param shift1: Number of indices to translate to the left along first axis
        
        :param shift2: Number of indices to translate downward along second axis
        '''
        
        if old_vec is None:
            old_vec = self.variable_occs
        
        n_var = len(self.variable_atoms)
        new_vec = np.zeros(n_var)
        for var_ind in range(n_var):
            d1, d2 = self.var_ind_to_sym_inds(var_ind)
            map_from_ind = self.sym_inds_to_var_ind( d1 + shift1 , d2 + shift2 )
            new_vec[var_ind] = old_vec[ map_from_ind ]
                
        return new_vec
        
    
    def generate_all_translations_and_rotations(self, old_vec = None):
        
        if old_vec is None:
            old_vec = self.variable_occs
        
        all_symmetries = []
        n_var = len(self.variable_atoms)
        
        for var_ind in range(n_var):
            d1, d2 = self.var_ind_to_sym_inds(var_ind)
            translated = self.translate( d1, d2, old_vec = old_vec)
            
            for angle in [0,1,2]:
                all_symmetries.append( self.rotate( angle, old_occs = translated ) )

        return np.array(all_symmetries)
    
    
    def rotate(self, i, old_occs = None):
        '''
        Permute occupancies according to symmetry
        
        :param i: Rotate the catalyst by 120i degrees (clockwise or counterclockwise?)
        '''
        
        if old_occs is None:
            old_occs = self.variable_occs
        
        i = i%3
        
        if i == 0:
            return old_occs
        else:
        
            n_var = len(self.variable_atoms)
            new_occs = np.zeros(n_var)
            for var_ind in range(n_var):
            
                d1, d2 = self.var_ind_to_sym_inds(var_ind)
                
                if i == 1:
                    d1_new = -d1 - d2
                    d2_new = d1
                elif i == 2:
                    d1_new = d2
                    d2_new = -d1 - d2
                else:
                    raise NameError('Error in rotations.')
                
                map_from_ind = self.sym_inds_to_var_ind( d1_new , d2_new )
                new_occs[var_ind] = old_occs[ map_from_ind ]
                    
            return new_occs

  
    def get_local_inds(self, shift1=0, shift2=0):
        '''
        Permute occupancies according to symmetry
        
        :param shift1: Number of indices to translate along first axis
        
        :param shift2: Number of indices to translate along second axis
        '''
        
        n_var = len(self.variable_atoms)
        new_occs = []
        for d2 in range(-3,4):
            for d1 in range(-3,4):
                map_from_ind = self.sym_inds_to_var_ind( d1 + shift1 , d2 + shift2 )
                new_occs.append(map_from_ind)
                
        return new_occs
        
    
    def sym_inds_to_var_ind(self, sym_ind1, sym_ind2):
        '''
        Convert 2-indices to 1
        :param sym_ind1: Index of the atom along the first dimension
        :param sym_ind2: Index of the atom along the second dimension
        :returns: Index of the atom
        '''
        sym_ind1 = sym_ind1 % self.dim1
        sym_ind2 = sym_ind2 % self.dim2
        return sym_ind2 * self.dim1 + sym_ind1
    
    
    def var_ind_to_sym_inds(self,var_ind):
        '''
        Convert 2-indices to 1
        :param: Index of the atom
        :returns sym_ind1: Index of the atom along the first dimension
        :returns sym_ind2: Index of the atom along the second dimension
        '''
        return var_ind % self.dim1, var_ind / self.dim1
        
        
    def show(self, fname = 'structure_1', fmat = 'png'):
                
        '''
        Print image of surface
        :param fname:   File name to which the file type will be appended
        :param fmat:    Format, png, xsd, or povray
        :param transmute_top:   Whether to turn the top layer into Co for easier visualization
        :param chop_top: Removes the top layer of atoms
        '''        
        
        # Build ASE atoms object from defected graph
        if self.atoms_defected is None:
            self.occs_to_atoms()
        
        self.atoms_defected.set_pbc(True)
        
        # Remove all Cu atoms
        chem_symbs = self.atoms_defected.get_chemical_symbols()
        if_delete = [False for atom in self.atoms_defected]
        
        for atom_ind in range(len(self.atoms_defected)):
            if chem_symbs[atom_ind] == 'Cu':
                if_delete[atom_ind] = True
        
        del self.atoms_defected[if_delete]
                
        
        if fmat == 'png':
            write(fname + '.png', self.atoms_defected )
        elif fmat == 'xsd':
            self.atoms_defected.set_pbc(True)
            write(fname + '.xsd', self.atoms_defected, format = fmat )
        elif fmat == 'povray':
            write(fname + '.pov', self.atoms_defected )
        else:
            raise NameError('wrong format to show catalyst')
    
    
    def calc_weights(self):
        '''
        Computes weights for distance metric
        :returns: Weight vector
        '''
        w = np.zeros(len(self.variable_occs))
        d = self.dim1
        ref_site_images = [[0,0],[0,d],[d,0],[d,d],[d,-d]]
        
        for i in xrange(len(self.variable_occs)):
            f1,f2 = self.var_ind_to_sym_inds(i)
            min_moves = []
            for image in ref_site_images:
                diff = [f1 - image[0], f2 - image[1], 0]
                scores = np.zeros(3)
                scores[0] = np.abs(diff[0]) + np.abs(diff[1])
                scores[1] = np.abs(diff[0]+diff[1]) + np.abs(-diff[0])
                scores[2] = np.abs(diff[0]+diff[1]) + np.abs(diff[1])
                min_moves.append(np.min(scores))
            
            nn_shell = min(min_moves)   # nearest neighbor shell that it is in
            w[i] = np.exp(-2*nn_shell)
        return w
        
    
def dyn_cat_dist(occs1,occs2,w):
    '''
    Distance metric between two defected catalyst sites
    :param occs1: Ni site occupancies of first catalyst
    :param occs2: Ni site occupancies of second catalyst
    :param w: weights for each of the site occupancy differences
    '''
    
    return np.dot( np.abs(occs1 - occs2), w )