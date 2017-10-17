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
    print fldr_name
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
    
    d_cut = 0.1     # Matches are not found if we lower this to 0.01
    
    mol_cart_coords = cat.atoms_template.get_positions()[:, 0:2:]
    
    for i in range(len(x.lat.site_type_inds)):
    
        lat_site_coords = x.lat.cart_coords[i, :]
        match_found = False
        
        
        defect_ind = -1
        while not match_found:
            
            defect_ind += 1
            
            if defect_ind >= len(cat.variable_atoms):
                raise NameError('Molecular site not found for lattice site.')
        
            atom_ind = cat.variable_atoms[defect_ind]
            atom_pos = mol_cart_coords[atom_ind,:]
            
            match_found = ( np.linalg.norm( atom_pos - lat_site_coords ) < d_cut )
            
        
        site_props_ss[defect_ind,:] = x.TS_site_props_ss[i,:]    
         
    
    '''
    Check if rate is correct
    '''
    
    rate_specnum = - (x.specnumout.spec[-1,2] - x.specnumout.spec[0,2]) / ( x.specnumout.t[-1] - x.specnumout.t[0] )
    rate_props = ( x.propCounter[-1,0] - x.propCounter[0,0] ) / ( x.specnumout.t[-1] - x.specnumout.t[0] )
    rate_siteprops = np.sum( site_props_ss[:,0] - site_props_ss[:,1] )
    
    print '\n'
    print fldr_name
    print 'From spec_num:\t' + str(rate_specnum)
    print 'From propensities:\t' + str(rate_props)
    print 'From site props:\t' + str(rate_siteprops)
    
    return site_props_ss