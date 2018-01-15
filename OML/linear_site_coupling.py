'''
Analytical solution to the linear site coupling model
'''

import numpy as np

def compute_site_rates_lsc(kmc_lat):
    '''
    Compute analytical solution to the site rates for a certain catalyst
    :param kmc_lat: KMC lattice
    :returns: site rates
    '''
    
    # Rate constants
    k_ads_terrace = 1.0
    k_des_terrace = 1.0
    k_diff = 1.0
    k_des_edge = 1.0
    
    # Build matrices
    
    kmc_lat.site_type_inds
    kmc_lat.neighbor_list
    n_sites = len(kmc_lat.site_type_inds)
    
    # If there are no sites
    if n_sites == 0:
        return np.array([])
    
    A = np.zeros([n_sites,n_sites])
    b = np.zeros(n_sites)
    A_rate = np.zeros([n_sites,n_sites])
    
    for site_ind in range(n_sites):
        
        # Adsorption/desorption
        if kmc_lat.site_type_inds[site_ind] == 1:
            A[site_ind,site_ind] += -1 * (k_ads_terrace + k_des_terrace)
            b[site_ind] += -k_ads_terrace
        elif kmc_lat.site_type_inds[site_ind] == 2:
            A[site_ind,site_ind] += -1 * k_des_edge
            A_rate[site_ind,site_ind] = k_des_edge
        
        # Diffusion
        for site_ind_2 in range(n_sites):
        
            if [site_ind,site_ind_2] in kmc_lat.neighbor_list or [site_ind_2,site_ind] in kmc_lat.neighbor_list:
                A[site_ind,site_ind] += -1 * k_diff
                A[site_ind,site_ind_2] += k_diff
    
    # Solve linear system of equations for the coverages
    theta = np.linalg.solve(A, b)
    return A_rate.dot(theta)