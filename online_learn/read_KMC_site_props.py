import numpy as np
import os

import zacros_wrapper as zw
from NH3.NiPt_NH3_simple import NiPt_NH3_simple




def read_x(fldr_name):

    '''
    Read structure occupancies from a KMC folder
    '''

    cat = NiPt_NH3_simple()
    cat.load_defects(os.path.join(fldr_name,'structure.xsd'))
    return cat.variable_occs
    
    
def read_y(fldr_name):

    '''
    Read site propensities from a KMC folder
    '''
    
    x = zw.kmc_traj()
    x.Path = fldr_name
    x.ReadAllOutput(build_lattice=True)
    
    n_rxns = len( x.genout.RxnNameList )
    cat = NiPt_NH3_simple()
    cat.load_defects(os.path.join(fldr_name,'structure.xsd'))
    site_props_ss = np.zeros( [cat.atoms_per_layer, n_rxns] )
    
    '''
    Fill in the rows of atoms that are present
    '''
    
    d_cut = 0.01
    
    for i in range(len(cat.variable_atoms)):
        
        atom_ind = cat.variable_atoms[i]
        
        # Get position and atomic number of the template atom we are trying to find
        cart_coords = cat.atoms_template.get_positions()[atom_ind, 0:2:]
        
        defect_ind = -1      # index of defect atom which might be a match
        dist = 1.0      # distance between atoms we are trying to match
        
        match_found = False
        
        while (not match_found) and defect_ind < len(x.lat.site_type_inds)-1:
        
            defect_ind += 1
            defect_coords = x.lat.cart_coords[defect_ind, :]
            dist = np.linalg.norm( cart_coords - defect_coords )
            match_found = (dist < d_cut)                # Second condition checks whether the elements match
        
        if match_found:
            site_props_ss[i,:] = x.TS_site_props_ss[defect_ind,:]
    
    return site_props_ss