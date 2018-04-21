import numpy as np

from ORR import *

cat = orr_cat()
cat.randomize(coverage = 0.7, build_structure = True)

cat.show('before_annealed', fmat = 'png', chop_top = False)

passes = 5

for i in range(passes):                                                             # Do multiple sweeps of the surface
    for j in cat.variable_atoms:                                                        # For each variable atom    
        if cat.defected_graph.is_node(j):                                           # If it is occupied...
            CN_list = [ cat.defected_graph.get_coordination_number(j) ]             # Start making coordination list
            spot_list = [ j ]
            for neighb in cat.variable_atoms:
            #for neighb in cat.template_graph.get_neighbors(j):                      # For each neighbor around 
                if not cat.defected_graph.is_node(neighb):                          # If the neighbor is empty
                    # compute its CN
                    if neighb in cat.template_graph.get_neighbors(j):
                        CN = -1         # Compensate for the fact that the original site has an atom
                    else:
                        CN = 0
                    for neighb_2 in cat.template_graph.get_neighbors(neighb):
                        if cat.defected_graph.is_node(neighb_2):
                            CN += 1
                    CN_list.append(CN)
                    spot_list.append(neighb)
            
            go_to = 0
            best_CN = CN_list[0]
            for choice in range(1, len(spot_list)):
                if CN_list[choice] > best_CN:
                    go_to = choice
                    best_CN = CN_list[choice]
            print go_to    
            if not go_to == 0:
                cat.flip_atom(j)    
                cat.flip_atom(spot_list[go_to])
                    
                
cat.occs_to_atoms() 
cat.show('rand_annealed', fmat = 'png', chop_top = False) 
        
np.save('X.npy', cat.variable_occs)