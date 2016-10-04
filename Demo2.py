# -*- coding: utf-8 -*-
"""
Created on Tue Oct 04 12:53:53 2016

@author: mpnun
"""

# Build bridge sites

import os
import sys
import numpy as np
from ase.io import read, write, vasp
from ase import Atoms

from Cat_structure import Cat_structure
from Helper import Helper

os.system('cls')

POSCAR_fname = 'graphene.POSCAR'
POSCAR_out_fname = 'graphene_with_bridge.xyz'

''' Prepare bare catalyst structure '''
IGO = Cat_structure()
mol_dat = read(POSCAR_fname)
IGO.bare_cat = mol_dat


''' Find bridge sites '''
bridge_sites = Helper.find_bridge(IGO.bare_cat)
IGO.bare_cat.extend(bridge_sites)

write(POSCAR_out_fname, IGO.bare_cat, format='xyz')