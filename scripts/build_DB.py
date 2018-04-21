'''
Test evlaution of the linear site coupling model on some test structures
'''

import sys
sys.path.append('/home/vlachos/mpnunez/Github/Zacros-Wrapper')
sys.path.append('/home/vlachos/mpnunez/Github/Structure-Optimization')
sys.path.append('/home/vlachos/mpnunez/python_packages/ase')
sys.path.append('/home/vlachos/mpnunez/python_packages/sklearn/lib/python2.7/site-packages')

import os
import numpy as np
import matplotlib as mat
import matplotlib.pyplot as plt

import zacros_wrapper as zw
import OML

n_structure = 10000

occ_DB = None
KMC_site_type_DB = None
site_rates_DB = None

# Generate all symmetry indeices
cat = OML.LSC_cat()
site_inds = range(len(cat.variable_atoms))
site_ind_symms = cat.generate_all_translations_and_rotations(old_vec = site_inds)
site_ind_symms = site_ind_symms.astype(int)

for struc_ind in xrange(n_structure):

    print struc_ind
    # Make a random structure
    cat = OML.LSC_cat()
    cat.randomize(build_structure = True)
    #cat.show(fname = 'test_struc')
    
    '''
    Ni site occupancies
    '''
    occ_syms = np.zeros([3*len(cat.variable_occs),len(cat.variable_occs)])
    for i in range(3*len(cat.variable_occs)):
        occ_syms[i,:] = np.array(cat.variable_occs)[site_ind_symms[i,:]]
    
    if occ_DB is None:
        occ_DB = occ_syms
    else:
        occ_DB = np.vstack([occ_DB,occ_syms])
    
    '''
    KMC site types
    '''
    cat.graph_to_KMClattice()
    kmc_site_types = np.zeros(len(cat.variable_occs))
    for i in range(len(cat.KMC_lat.site_type_inds)):
        kmc_site_types[cat.var_ind_kmc_sites[i]] = cat.KMC_lat.site_type_inds[i]
    
    kmc_st_allsyms = np.zeros([3*len(cat.variable_occs),len(cat.variable_occs)])
    for i in range(3*len(cat.variable_occs)):
        kmc_st_allsyms[i,:] = np.array(kmc_site_types)[site_ind_symms[i,:]]
    
    if KMC_site_type_DB is None:
        KMC_site_type_DB = kmc_st_allsyms
    else:
        KMC_site_type_DB = np.vstack([KMC_site_type_DB,kmc_st_allsyms])
    
    
    '''
    Compute rates
    '''
    site_rates = cat.compute_site_rates_lsc()
        
    if site_rates_DB is None:
        site_rates_DB = site_rates
    else:
        site_rates_DB = np.hstack([site_rates_DB,site_rates])

# Triplicate site rates to account for rotations
y = np.tile(site_rates_DB,[3,1])
site_rates_DB = np.transpose(y).flatten()

# Save database files
np.save('occ_DB.npy', occ_DB.astype(int))       # make sure to store this as ints
np.save('KMC_site_type_DB.npy', KMC_site_type_DB.astype(int))
np.save('site_rates_DB.npy', site_rates_DB)