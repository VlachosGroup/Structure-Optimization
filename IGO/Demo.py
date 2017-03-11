# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 12:44:43 2016

@author: mpnun
"""

import os
import sys
import numpy as np
from ase.io import read, write, vasp
from ase import Atoms

sys.path.insert(0, '../Core')
from Cat_structure import Cat_structure
from Helper import Helper
from Machine_learning import Machine_learning

os.system('cls')

POSCAR_fname = 'graphene.POSCAR'
POSCAR_out_fname = 'optimized.xyz'

''' Prepare bare catalyst structure '''
IGO = Cat_structure()
mol_dat = read(POSCAR_fname)
IGO.bare_cat = mol_dat

# Find bridge sites and define site types
bridge_sites = Helper.find_bridge(IGO.bare_cat)
IGO.site_types = [0 for i in range(IGO.bare_cat.positions.shape[0])]
for i in range(bridge_sites.positions.shape[0]):
    IGO.site_types.append(1)
IGO.site_locs = np.vstack([IGO.bare_cat.positions, bridge_sites.positions])

''' Add functional groups ''' 
hydroxyl_up = Atoms('OH', positions=[(0, 0, 0.5),(0, 0, 1.0)])       # C-OH
hydroxyl_down = Atoms('OH', positions=[(0, 0, -0.5),(0, 0, -1.0)])       # C-OH
epoxy_up = Atoms('O', positions=[(0, 0, 0.3)])                       # C-O-C, # May be epoxy (C-C bond present) or ether (C-O-C with no C-C bond)
epoxy_down = Atoms('O', positions=[(0, 0, -0.3)])                       # C-O-C
carbonyl_up = Atoms('O', positions=[(0, 0, 0.5)])                    # C=O
carbonyl_down = Atoms('O', positions=[(0, 0, -0.5)])                    # C=O

IGO.functional_group_list = [hydroxyl_up, hydroxyl_down, epoxy_up, epoxy_down, carbonyl_up, carbonyl_down]
IGO.func_groups_allowed = [[0,1,2,5,6], [0,3,4]]

''' Build IGO site list '''
#IGO.randomize_occs()           # put random functionalities
IGO.site_occs = [0 for i in range(len(IGO.site_types))]
IGO.seed_functional_groups(0, [33,8,7,0,0,24,24])
IGO.seed_functional_groups(1, [129,0,0,4,3,0,0])

#IGO.functionalize()
IGO.find_neighbs()
#
## Set up and run optimization
ML = Machine_learning()
ML.cat_structure = IGO
ML.patterns = [(0,0), (5,5), (6,6), (5,1), (6,2), (1,1), (2,2), (3,3), (4,4)]
ML.pattern_engs = [1,1,1,1,1,1,1,1,1]
ML.Metropolis()
ML.PlotTrajectory()

# Print out optimum
ML.cat_structure.functionalize()
write(POSCAR_out_fname, ML.cat_structure.functional_cat, format='xyz')