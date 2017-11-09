# -*- coding: utf-8 -*-
"""
Created on Fri May 26 13:26:51 2017

@author: mpnun
"""

import numpy as np
import os
from matplotlib.mlab import PCA

class metal:

    '''
    Class for properties of a metal
    '''
    
    def __init__(self, met_name):
        
        self.name = met_name
        
        if met_name == 'Pt':
            
            self.E_coh = 4.5222 				 
            self.lc_PBE = 3.968434601
            self.GCN_opt = 8.3
            self.E_per_bulk_atom = -6.0978575
            self.compute_UQ_params('Pt_BEs.npy')
			
        elif met_name == 'Au':
            
            self.E_coh = 2.3645 				 
            self.lc_PBE = 4.155657928
            self.GCN_opt = 5.8				
            self.E_per_bulk_atom = -3.2201925
            self.compute_UQ_params('Au_BEs.npy')
            
        else:
            
            raise ValueError(met_name + ' not found in metal database.')
        
    
    def compute_UQ_params(self,np_fname):
        '''
        :param np_fname: Name of the numpy file with binding energy data
        '''
        dir = os.path.dirname(__file__)
        np_fname = os.path.join(dir, np_fname)
        BEs = np.load(np_fname)
    
        # Regress OH BE vs. GCN
        self.OH_slope, self.OH_int = np.polyfit(BEs[:,0], BEs[:,1], 1)
        BE_OH_pred = BEs[:,0] * self.OH_slope + self.OH_int
        res_OH = BEs[:,1] - BE_OH_pred          # Compute residuals
        self.sigma_OH_BE = np.std(res_OH)       # Compute variance of residuals

        # Regress OOH BE vs. GCN
        self.OOH_slope, self.OOH_int = np.polyfit(BEs[:,0], BEs[:,2], 1)
        BE_OOH_pred = BEs[:,0] * self.OOH_slope + self.OOH_int
        res_OOH = BEs[:,2] - BE_OOH_pred         # Compute residuals
        self.sigma_OOH_BE = np.std(res_OOH)     # Compute variance of residuals
        
        '''
        Perform PCA on residuals
        '''
        data = np.transpose( np.vstack([res_OH, res_OOH]) )
        eigenvectors, eigenvalues, V = np.linalg.svd(data.T, full_matrices=False)
        projected_data = np.dot(data, eigenvectors)
        sigma = projected_data.std(axis=0)
        self.pca_mat = eigenvectors
        self.sigma_pca_1 = sigma[0]
        self.sigma_pca_2 = sigma[1]
            
    
    def get_BEs(self, GCN, uncertainty = False, correlations = True):
        '''
        :param GCN: generalized binding energy of the site
        :param uncertainty: If true, add random noise due to error in GCN relation
        :param correlations: If true, use PCA as joint PDF of 
        '''
        
        OH_BE = self.OH_slope * GCN + self.OH_int
        OOH_BE = self.OOH_slope * GCN + self.OOH_int
        
        if uncertainty:
            
            if correlations:
            
                pca1 = self.sigma_pca_1 * np.random.normal()
                pca2 = self.sigma_pca_2 * np.random.normal()
                BE_errors = np.matmul(np.array([pca1, pca2]), self.pca_mat )
                OH_BE_error = BE_errors[0]
                OOH_BE_error = BE_errors[1]
                
            else:
            
                OH_BE_error = self.sigma_OH_BE * np.random.normal()
                OOH_BE_error = self.sigma_OOH_BE * np.random.normal()
            
            OH_BE += OH_BE_error
            OOH_BE += OOH_BE_error
        
        return [OH_BE, OOH_BE]