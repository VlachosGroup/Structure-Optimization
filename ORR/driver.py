# -*- coding: utf-8 -*-
"""
Created on Fri May 26 13:51:00 2017

@author: mpnun
"""

import os
from Cat_structure import cat_structure
import random


os.system('cls')

x = cat_structure('Pt', '111', 12, 12)
random.seed(9999)
x.seed_occs()
x.optimize(omega = 0.7, n_cycles = 500, n_record = 501, n_snaps = 26)
#x.optimize(ensemble = 'CE', omega = 0, n_cycles = 1, n_record = 101, n_snaps = 6)