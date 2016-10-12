# -*- coding: utf-8 -*-
"""
Created on Thu Oct 06 17:51:14 2016

@author: mpnun
"""

import os
from random import shuffle

from Cat_structure import Pattern

os.system('cls')

#p = Pattern(ads_types = [1,2,3], neighbs = [[1,2],[2,3],[1,3]])
#
#print p.ads_types
#print p.adj_mat

site_inds = range(20)
occs = [5,8,7]
shuffle(site_inds)


list_of_lists = []
ind = 0
for i in occs:
    sub_occs = site_inds[ind:ind+i]
    list_of_lists.append(sub_occs)
    ind += i
    
print list_of_lists