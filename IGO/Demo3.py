# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 12:44:43 2016

@author: mpnun
"""

import os
import sys

from ase.io import read, write, vasp
from ase import Atoms

sys.path.insert(0, '../Core')
from Cat_structure_GT import Cat_structure_GT
from Helper import Helper
from Machine_learning import Machine_learning

os.system('cls')

POSCAR_fname = 'graphene.POSCAR'
POSCAR_out_fname = 'optimized.xyz'

# Read bare catalyst structure
IGO = Cat_structure_GT()
IGO.bare_cat = read(POSCAR_fname)

# Find bridge sites and set site_cat
bridge_sites = Helper.find_bridge(IGO.bare_cat)
IGO.site_cat = IGO.bare_cat.copy().extend(bridge_sites)
#write('top_and_bridge.xyz', IGO.site_cat, format='xyz')

# Convert molecular site object to a graph
element_to_sitetype = {'C': 'top', 'Xe': 'bridge'}
IGO.ASE_to_graph(element_to_sitetype)
#IGO.show_graph()

# Define functional groups
empty = Atoms()
hydroxyl_up = Atoms('OH', positions=[(0, 0, 0.5),(0, 0, 1.0)])       # C-OH
hydroxyl_down = Atoms('OH', positions=[(0, 0, -0.5),(0, 0, -1.0)])       # C-OH
epoxy_up = Atoms('O', positions=[(0, 0, 0.3)])                       # C-O-C, # May be epoxy (C-C bond present) or ether (C-O-C with no C-C bond)
epoxy_down = Atoms('O', positions=[(0, 0, -0.3)])                       # C-O-C
carbonyl_up = Atoms('O', positions=[(0, 0, 0.5)])                    # C=O
carbonyl_down = Atoms('O', positions=[(0, 0, -0.5)])                    # C=O
IGO.functional_group_dict = {'empty': empty, 'hydroxyl_up': hydroxyl_up, 'hydroxyl_down': hydroxyl_down, 'epoxy_up': epoxy_up, 'epoxy_down': epoxy_down, 'carbonyl_up': carbonyl_up, 'carbonyl_down': carbonyl_down}

# Populate functional groups
IGO.allowed_groups = {'top': ['empty', 'hydroxyl_up', 'hydroxyl_down', 'carbonyl_up', 'carbonyl_down'], 'bridge': ['empty', 'epoxy_up', 'epoxy_down']}
#IGO.randomize_occs()
IGO.seed_func_groups(8, 'hydroxyl_up', 'top')
IGO.functionalize()
#write('functionalized.xyz', IGO.functional_cat, format='xyz')