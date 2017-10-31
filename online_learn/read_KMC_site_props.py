import numpy as np
import os

import zacros_wrapper as zw
from NH3.NiPt_NH3_simple import NiPt_NH3_simple
    
from multiprocessing import Pool
    
def read_many_calcs(fldr_list):

    # Run in parallel
    pool = Pool()
    kmc_data_list = pool.map(read_occs_and_rates, fldr_list)
    pool.close()
    
    structure_occs = []
    site_rates = []    
    for kmc_data in kmc_data_list:
        structure_occs.append(kmc_data[0])
        site_rates.append(kmc_data[1])
    
    structure_occs = np.array(structure_occs)
    site_rates = np.array(site_rates)

    return [ structure_occs , site_rates ]
    
def read_occs_and_rates(fldr_name, gas_prod = 'A', gas_stoich = -1):

    '''
    Read structure occupancies and site propensities from a KMC folder
    '''
    
    print fldr_name
    
    x = zw.kmc_traj()
    x.Path = fldr_name
    x.ReadAllOutput(build_lattice=True)
    
    n_rxns = len( x.genout.RxnNameList )
    cat = NiPt_NH3_simple()
    cat.load_defects(os.path.join(fldr_name,'structure.xsd'))
    site_props_ss = np.zeros( [cat.atoms_per_layer, n_rxns] )
    
    '''
    Identify stoichiometries
    '''
    
    # Find index of product molecule
    try:
        gas_prod_ind = len( x.simin.surf_spec ) + x.simin.gas_spec.index( gas_prod )           # Find the index of the product species and adjust index to account for surface species
    except:
        raise Exception('Gas species ' + gas_prod + ' not found.')

    # Find the stochiometry of the product molecule for each reaction
    nRxns = len(x.genout.RxnNameList)
    TOF_stoich = np.zeros(nRxns)
    for i, elem_stoich in enumerate(x.genout.Nu):
        TOF_stoich[i] = elem_stoich[gas_prod_ind]
    
    
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
                print lat_site_coords
                raise NameError('Molecular site not found for lattice site.')
        
            atom_ind = cat.variable_atoms[defect_ind]
            atom_pos = mol_cart_coords[atom_ind,:]
            
            match_found = ( np.linalg.norm( atom_pos - lat_site_coords ) < d_cut )
            
        
        site_props_ss[defect_ind,:] = x.TS_site_props_ss[i,:]    
    
    site_rates = np.matmul( site_props_ss, TOF_stoich ) / gas_stoich
    
    return [cat.variable_occs, site_rates]