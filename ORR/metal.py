# -*- coding: utf-8 -*-
"""
Created on Fri May 26 13:26:51 2017

@author: mpnun
"""

class metal:
    
    name = None
    E_coh      = None     
    lc_PBE = None
    fccnndist = None
    OHslope = None
    OHint = None
    OOHslope = None
    OOHint = None
    GCN_opt = None
    E_per_bulk_atom  = None       # from DFT
    
    def __init__(self, met_name):
        
        self.name = met_name
        
        if met_name == 'Pt':
            
            self.E_coh = 4.5222 				 
            self.lc_PBE = 3.968434601				
            self.OHslope = 0.1915   	  
            self.OHint = -3.8166     	  
            self.OOHslope = 0.1697   		    
            self.OOHint = -2.3411			
            self.GCN_opt = 8.3
            self.E_per_bulk_atom = -6.0978575
            
        elif met_name == 'Au':
            
            self.E_coh = 2.3645 				 
            self.lc_PBE = 4.155657928				
            self.OHslope = 0.1163   	  
            self.OHint = -3.0926     	  	  
            self.OOHslope = 0.1163   		    		    
            self.OOHint = -1.4081					
            self.GCN_opt = 5.8				
            self.E_per_bulk_atom = -3.2201925
            
        else:
            
            raise ValueError(met_name + ' not found in metal database.')
        
        
    def get_OH_BE(self, GCN):       # Can give this an option to produce random OH and OOH BEs with correlations
        return self.OHslope * GCN + self.OHint
        
        
    def get_OOH_BE(self, GCN):
        return self.OOHslope * GCN + self.OOHint