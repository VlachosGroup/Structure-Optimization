# -*- coding: utf-8 -*-
"""
Created on Thu Oct 06 17:51:14 2016

@author: mpnun
"""

import os

from Cat_structure import Pattern

os.system('cls')

p = Pattern(ads_types = [1,2,3], neighbs = [[1,2],[2,3],[1,3]])

print p.ads_types
print p.adj_mat