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

from Cat_structure import Cat_structure

os.system('cls')

POSCAR_fname = 'graphene.POSCAR'
POSCAR_out_fname = 'functionalized.xyz'

''' Prepare bare catalyst structure '''
IGO = Cat_structure()
mol_dat = read(POSCAR_fname)
IGO.bare_cat = mol_dat

''' Add functional groups '''       # need to check these
hydroxyl_up = Atoms('OH', positions=[(0, 0, 0.5),(0, 0, 1.0)])       # C-OH
hydroxyl_down = Atoms('OH', positions=[(0, 0, -0.5),(0, 0, -1.0)])       # C-OH
epoxy_up = Atoms('O', positions=[(0, 0, 0.5)])                       # C-O-C
epoxy_down = Atoms('O', positions=[(0, 0, -0.5)])                       # C-O-C
carbonyl_up = Atoms('O', positions=[(0, 0, 0.3)])                    # C=O
carbonyl_down = Atoms('O', positions=[(0, 0, -0.3)])                    # C=O

IGO.functional_group_list = [hydroxyl_up, hydroxyl_down, epoxy_up, epoxy_down, carbonyl_up, carbonyl_down]

''' Build IGO site list '''
IGO.site_locs = IGO.bare_cat.positions
IGO.site_occs = np.random.randint(low = 0, high = 5, size=IGO.site_locs.shape[0])           # put random functionalities
IGO.functionalize()

write(POSCAR_out_fname, IGO.functional_cat, format='xyz')
#vasp.write_vasp('POSCAR', IGO.functional_cat, vasp5=True, direct=True)

IGO.find_neighbs()

print IGO.count_pairs(1, 1)
print IGO.count_pairs(1, 3)