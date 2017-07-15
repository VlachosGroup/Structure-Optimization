# -*- coding: utf-8 -*-
"""
Created on Fri May 26 13:51:00 2017

@author: mpnun
"""

import os
from Cat_structure import cat_structure
import random


import sys
sys.path.append('C:\Users\mpnun\Dropbox\Github\Zacros-Wrapper')
import zacros_wrapper as zw


os.system('cls')

# Create catalyst structure
x = cat_structure('Pt', '111', 12, 12)
random.seed(9999)
x.randomize()

# Read KMC input file template
kmc = zw.kmc_traj(path = 'C:\Users\mpnun\Dropbox\DesignActiveSites\ORR\KMC')
kmc.ReadAllInput()

# Clear cluster and energetics variants
for cluster in kmc.clusterin.cluster_list:
    cluster.variant_list = []
    
for reaction in kmc.mechin.rxn_list:
    reaction.variant_list = []



# KMC lattice
kmc.lat = zw.Lattice()
kmc.lat.text_only = False
kmc.lat.lattice_matrix = x.atoms_obj_template.get_cell()[0:2, 0:2]
cart_coords_list = []


# Loop through active sites
site_ind = 0
for i in x.active_atoms:
    if x.defected_graph.is_node(i):
        if x.defected_graph.get_coordination_number(i) <= 9:
            
            site_name = 'site_' + str(site_ind+1)
            
            kmc.lat.site_type_names.append(site_name)
            kmc.lat.site_type_inds.append(site_ind+1)
            cart_coords_list.append( x.atoms_obj_template.get_positions()[i, 0:2:] )
            
            gcn = x.defected_graph.get_generalized_coordination_number(i, 12)
            BE_OH = x.metal.get_OH_BE(gcn)
            BE_OOH = x.metal.get_OOH_BE(gcn)
            
            '''
            Add reaction variants to mechanism_input.dat
            '''
            
            # O2(g) -> OOH*
            rv = zw.rxn_variant()
            rv.name = site_name
            rv.site_types = [site_name]             # site types
            rv.pre_expon = 1.0e13              # pre-exponential factor
            rv.pe_ratio = 1.0                # partial equilibrium ratio
            rv.activ_eng = 0.0                # activation energy
            rv.prox_factor = 0.5              # proximity factor
            kmc.mechin.rxn_list[0].variant_list.append(rv)
            
            # OOH* -> O*
            rv = zw.rxn_variant()
            rv.name = site_name
            rv.site_types = [site_name]              # site types
            rv.pre_expon = 1.0e13              # pre-exponential factor
            rv.pe_ratio = 1.0                # partial equilibrium ratio
            rv.activ_eng = 0.0                # activation energy
            rv.prox_factor = 0.5              # proximity factor
            kmc.mechin.rxn_list[1].variant_list.append(rv)
            
            # O* -> OH*
            rv = zw.rxn_variant()
            rv.name = site_name
            rv.site_types = [site_name]              # site types
            rv.pre_expon = 1.0e13              # pre-exponential factor
            rv.pe_ratio = 1.0                # partial equilibrium ratio
            rv.activ_eng = 0.0                # activation energy
            rv.prox_factor = 0.5              # proximity factor
            kmc.mechin.rxn_list[2].variant_list.append(rv)
            
            # OH* -> H2O(l)
            rv = zw.rxn_variant()
            rv.name = site_name
            rv.site_types = [site_name]              # site types
            rv.pre_expon = 1.0e13              # pre-exponential factor
            rv.pe_ratio = 1.0                # partial equilibrium ratio
            rv.activ_eng = 0.0                # activation energy
            rv.prox_factor = 0.5              # proximity factor
            kmc.mechin.rxn_list[3].variant_list.append(rv)
            
            '''
            Add clusters variant to energetics_input.dat
            '''
            
            # OOH* point
            cv = zw.cluster_variant()
            cv.name = site_name
            cv.site_types = [site_name]
            cv.graph_multiplicity = 1
            cv.cluster_eng = 0.0
            kmc.clusterin.cluster_list[0].variant_list.append(cv)
            
            # O* point
            cv = zw.cluster_variant()
            cv.name = site_name
            cv.site_types = [site_name]
            cv.graph_multiplicity = 1
            cv.cluster_eng = 0.0
            kmc.clusterin.cluster_list[1].variant_list.append(cv)
            
            # OH* point
            cv = zw.cluster_variant()
            cv.name = site_name
            cv.site_types = [site_name]
            cv.graph_multiplicity = 1
            cv.cluster_eng = 0.0
            kmc.clusterin.cluster_list[2].variant_list.append(cv)
            
            site_ind += 1


# Finish building the list
kmc.lat.set_cart_coords(cart_coords_list)
kmc.lat.Build_neighbor_list(cut = 2.87)    
        
# Print input files
kmc.Path = 'C:\Users\mpnun\Dropbox\DesignActiveSites\ORR\KMC\out_test'
kmc.WriteAllInput()
kmc.PlotLattice()