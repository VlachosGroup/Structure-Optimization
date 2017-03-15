# Use this for converting Wei's model

import sys
import os
import numpy as np
import matplotlib as mat
import matplotlib.pyplot as plt
import copy
import random
from shutil import copyfile
import pickle

sys.path.append('/home/vlachos/mpnunez/ase')
from ase.build import fcc100
from ase.io import read
from ase.visualize import view
from ase.io import write

import networkx as nx
import networkx.algorithms.isomorphism as iso

sys.path.append('/home/vlachos/mpnunez/Github/Zacros-Wrapper')
import zacros_wrapper.Lattice as lat
import zacros_wrapper as zw

from template import dyno_struc


class AB_model(dyno_struc):
    
    '''
    Handles a dynamic lattice for a toy chemistry
    '''
    
    def __init__(self):
        
        '''
        Call superclass constructor
        '''
        
        super(AB_model, self).__init__()
        
        
    def build_template(self):
    
        '''
        Build template ase atoms object, 1-layer of Pt(100)
        '''
    
        self.atoms_template = fcc100('Pt', size=(10, 10, 1), vacuum=15.0)


    
    def generate_defects(self, mode = 'rand_cov'):
    
        '''
        Create random defects in the structure
        '''
        
        self.atoms_defected = copy.deepcopy(self.atoms_template)
        
        a_nums = [78 for i in range(len(self.atoms_defected))]
        chem_symbs = ['Pt' for i in range(len(self.atoms_defected))]        
        
        if mode == 'rand_cov':          # random nickel coverage
            
            n_Ni = random.randint(1, 99)
            Ni_inds = random.sample(range(100), n_Ni)
            
            for Ni_ind in Ni_inds:
                a_nums[Ni_ind] = 28
                chem_symbs[Ni_ind] = 'Ni'
        
        else:                           # all sites randomized independently
        
            n_var = 10 * 10
            n_fixed = 0
            n_tot = n_var + n_fixed
            
            for i in range(n_fixed, n_tot):
                if random.uniform(0, 1) < 0.5:
                    a_nums[i] = 28
                    chem_symbs[i] = 'Ni'
                    
        self.atoms_defected.set_atomic_numbers(a_nums)
        self.atoms_defected.set_chemical_symbols(chem_symbs)
        
    
    def template_to_KMC_lattice(self):

        '''
        Convert defected atoms object to a KMC lattice object
        '''
    
        self.KMC_lat = lat()
        self.KMC_lat.workingdir = self.path

        self.KMC_lat.lattice_matrix = self.atoms_defected.get_cell()[0:2, 0:2]
        self.KMC_lat.repeat = [1,1]
        self.KMC_lat.site_type_names = ['Pt', 'Ni']
        self.KMC_lat.site_type_inds = [1 for i in range(len(self.atoms_defected))]
        
        chem_symbs = self.atoms_defected.get_chemical_symbols()

        for i in range(len(self.atoms_defected)):
            if chem_symbs[i] == 'Pt':
                self.KMC_lat.site_type_inds[i] = 1
            elif chem_symbs[i] == 'Ni':
                self.KMC_lat.site_type_inds[i] = 2
            else:
                raise ValueError('Unknown atom type')
        
        frac_coords = self.atoms_defected.get_scaled_positions(wrap=False)
        self.KMC_lat.frac_coords = frac_coords[:,0:2]   # project onto the x-y plane

        self.KMC_lat.cart_coords = self.atoms_defected.get_positions()
        self.KMC_lat.Build_neighbor_list()
        
        



    def generate_fingerprint_list(self):
        
        '''
        Generate a list of subgraphs to enumerate
        '''
        
        self.fingerprint_graphs = [nx.Graph() for i in range(5)]
        
        # subgraph 1
        self.fingerprint_graphs[0].add_nodes_from([1])
        self.fingerprint_graphs[0].add_edges_from([])
        self.fingerprint_graphs[0].node[1]['type'] = 'Pt'
        
        # subgraph 2
        self.fingerprint_graphs[1].add_nodes_from([1])
        self.fingerprint_graphs[1].add_edges_from([])
        self.fingerprint_graphs[1].node[1]['type'] = 'Ni'
        
        # subgraph 3
        self.fingerprint_graphs[2].add_nodes_from([1,2])
        self.fingerprint_graphs[2].add_edges_from([[1,2]])
        self.fingerprint_graphs[2].node[1]['type'] = 'Pt'
        self.fingerprint_graphs[2].node[2]['type'] = 'Pt'
        
        # subgraph 4
        self.fingerprint_graphs[3].add_nodes_from([1,2])
        self.fingerprint_graphs[3].add_edges_from([[1,2]])
        self.fingerprint_graphs[3].node[1]['type'] = 'Ni'
        self.fingerprint_graphs[3].node[2]['type'] = 'Ni'
        
        # subgraph 5
        self.fingerprint_graphs[4].add_nodes_from([1,2])
        self.fingerprint_graphs[4].add_edges_from([[1,2]])
        self.fingerprint_graphs[4].node[1]['type'] = 'Pt'
        self.fingerprint_graphs[4].node[2]['type'] = 'Ni'
        
    
    



    def show_all(self):
    
        '''
        Print lattice_input.dat file and other graphs
        '''
    
        # View slab in ASE GUI
#        write('ase_slab.png', self.atoms_defected)
        
        # Write the lattice_input.dat file
        self.KMC_lat.Write_lattice_input()
        
        # Graph the KMC lattice
        plt = self.KMC_lat.PlotLattice(plot_neighbs = True, ms = 9)
        plt.savefig(os.path.join(self.path, 'lattice.png'))
        plt.close()
        
        
        # Graph the networkx graph
        d = {}
        for i in range(len(self.KMC_lat.site_type_inds)):
            d[i] = self.KMC_lat.cart_coords[i,:]
        nx.draw(self.lat_graph, pos = d)
#        plt.draw()
        plt.savefig(os.path.join(self.path, 'graph.png'))
        plt.close()
        
        
if __name__ == "__main__":
    
    '''
    Generate data for a conceptual example
    '''
    
    kmc_source = '/home/vlachos/mpnunez/Github/Dynamic-Catalyst-Structure/ABfiles'
    run_fldr = '/home/vlachos/mpnunez/ML/test'
    n_jobs = 94  # change this back to 94
    n_manual = 6
    zw.ClearFolderContents(run_fldr)
    f_list = [[] for i in range(n_jobs + n_manual)]    
    
    for i in range(n_jobs):    
        
        x = AB_model()
        x.path = os.path.join(run_fldr, str(i))
        
        if not os.path.exists(x.path):
                os.makedirs(x.path)                
        
        x.build_template()
        x.generate_defects()
        x.template_to_KMC_lattice()
        x.KMC_lattice_to_graph()
        x.generate_fingerprint_list()
        x.count_fingerprints()
        x.show_all()
        
        copyfile(os.path.join(kmc_source, 'simulation_input.dat'), os.path.join(x.path, 'simulation_input.dat'))
        copyfile(os.path.join(kmc_source, 'mechanism_input.dat'), os.path.join(x.path, 'mechanism_input.dat'))
        copyfile(os.path.join(kmc_source, 'energetics_input.dat'), os.path.join(x.path, 'energetics_input.dat'))
        
        f_list[i] = x.fingerprint_counts
    

    
    xsd_dir = '/home/vlachos/mpnunez/ML/xsds'
    for i in range(n_manual):
        
        x = AB_model()
        x.path = os.path.join(run_fldr, str(n_jobs + i))
        
        if not os.path.exists(x.path):
            os.makedirs(x.path)        
        
        
        
        x.build_template()
        x.atoms_defected = read(os.path.join(xsd_dir, 's' + str(i+1) + '.xsd'), format = 'xsd')
        
        # Change unit cell to something easier to manage
        ucell = x.atoms_defected.get_cell()
        cart_coords = x.atoms_defected.get_positions()
        l1 = np.linalg.norm(ucell[0,:])
        l2 = np.linalg.norm(ucell[1,:])
        l3 = np.linalg.norm(ucell[2,:])
        new_cell = np.array([[l1, 0, 0], [0, l2, 0], [0, 0, l3]])
        x.atoms_defected.set_cell(new_cell, scale_atoms=True)
        
        x.template_to_KMC_lattice()
        x.KMC_lattice_to_graph()
        x.generate_fingerprint_list()
        x.count_fingerprints()
        x.show_all()
        
        copyfile(os.path.join(kmc_source, 'simulation_input.dat'), os.path.join(x.path, 'simulation_input.dat'))
        copyfile(os.path.join(kmc_source, 'mechanism_input.dat'), os.path.join(x.path, 'mechanism_input.dat'))
        copyfile(os.path.join(kmc_source, 'energetics_input.dat'), os.path.join(x.path, 'energetics_input.dat'))
        
        f_list[n_jobs + i] = x.fingerprint_counts

    
    f_list = np.array(f_list)
    with open(os.path.join(x.path, 'f_list.pickle'),'w') as f:
        pickle.dump(f_list, f)
        
    print f_list