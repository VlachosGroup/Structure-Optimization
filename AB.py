# Use this for converting Wei's model

import sys
import os
import numpy as np
import matplotlib as mat
import matplotlib.pyplot as plt
import copy
import random

from ase.build import fcc100
from ase.io import read
from ase.visualize import view
from ase.io import write
from ase import Atoms

import Core.Lattice as lat
import networkx as nx

class AB_model:
    
    def __init__(self):
        
        self.atoms_template = []            # will be an ASE atoms object when 
        
        self.occupancies = []               # occupancies of different atoms
        self.atoms_defected = []            # ASE atoms object, like atoms_template, but with atoms missing or transmuted
        
        self.KMC_lat = []                   # KMC lattice object
        self.lat_graph = []                 # networkx graph object
        
    def build_template(self):
    
        self.atoms_template = fcc100('Pt', size=(10, 10, 1), vacuum=15.0)

    ''' Take occupancies and use them to build a defected structure '''
    def generate_defected(self):    
        
        self.atoms_defected = copy.deepcopy(self.atoms_template)
        
        n_var = 10 * 10
        n_fixed = 0
        n_tot = n_var + n_fixed
        
        a_nums = [78 for i in range(len(self.atoms_defected))]
        chem_symbs = ['Pt' for i in range(len(self.atoms_defected))]        
        
        for i in range(n_fixed, n_tot):
            if random.uniform(0, 1) < 0.5:
                a_nums[i] = 28
                chem_symbs[i] = 'Ni'
                
        self.atoms_defected.set_atomic_numbers(a_nums)
        self.atoms_defected.set_chemical_symbols(chem_symbs)
        
    
    def template_to_KMC_lattice(self):
    
        self.KMC_lat = lat()

        self.KMC_lat.workingdir = '.'
        self.KMC_lat.lattice_matrix = self.atoms_defected.get_cell()[0:2, 0:2]
        self.KMC_lat.repeat = [1,1]
        self.KMC_lat.site_type_names = ['Pt', 'Ni']
        self.KMC_lat.site_type_inds = [1 for i in range(len(self.atoms_defected))]
        
        chem_symbs = self.atoms_defected.get_chemical_symbols()
        print chem_symbs
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
        #self.KMC_lat.Build_neighbor_list()
        
    def KMC_lattice_to_graph(self):
        
        pass
        
    def show_all(self):
    
        # View slab in ASE GUI
#        write('ase_slab.png', self.atoms_defected)
        
        # Write lattice_input.dat
        self.KMC_lat.Write_lattice_input()
        
        # Graph the lattice
        plt = self.KMC_lat.PlotLattice(plot_neighbs = True)
        plt.savefig(os.path.join('.', 'lattice.png'))
        plt.close()