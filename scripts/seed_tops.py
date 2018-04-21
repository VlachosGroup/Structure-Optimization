import numpy as np

from NH3 import *

cat = NiPt_NH3_simple()
#cat.randomize(coverage = 0.7, build_structure = True)
cat.variable_occs = [0 for i in range(cat.atoms_per_layer)]
cat.occs_to_atoms()
cat.show('before_annealed', fmat = 'png', chop_top = True)

n_seeds = 20
top_sites = np.random.choice( range(len(cat.variable_atoms)), size=n_seeds, replace=False )

for i in top_sites:
    
    cat.variable_occs[i] = 1
    sym_inds = cat.var_ind_to_sym_inds(i)
    
    ind1 = cat.sym_inds_to_var_ind(sym_inds[0] - 1, sym_inds[1] + 1)
    cat.variable_occs[ind1] = 1
    ind1 = cat.sym_inds_to_var_ind(sym_inds[0], sym_inds[1] + 1)
    cat.variable_occs[ind1] = 1
    ind1 = cat.sym_inds_to_var_ind(sym_inds[0] - 1, sym_inds[1])
    cat.variable_occs[ind1] = 1
    ind1 = cat.sym_inds_to_var_ind(sym_inds[0] + 1, sym_inds[1])
    cat.variable_occs[ind1] = 1
    ind1 = cat.sym_inds_to_var_ind(sym_inds[0], sym_inds[1] - 1)
    cat.variable_occs[ind1] = 1
    ind1 = cat.sym_inds_to_var_ind(sym_inds[0] + 1, sym_inds[1] - 1)
    cat.variable_occs[ind1] = 1
    
    
                    
                
cat.occs_to_atoms()
cat.show('rand_annealed', fmat = 'png', chop_top = True) 
        
np.save('X.npy', cat.variable_occs)