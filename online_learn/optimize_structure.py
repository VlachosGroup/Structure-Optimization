# Use the neural networks and optimize the structure

import numpy as np
import random
import time
import os



'''
Objective function
'''

def eval_rate(all_trans, nn_class, nn_pred, normalize_fac = 144):

    
    if all_trans is None:
        cat.variable_occs = sigma
        all_trans = cat.generate_all_translations()
    sites_are_active = nn_class.predict(all_trans)
    
    active_site_list = []
    for site_ind in range(len(sites_are_active)):
        if sites_are_active[site_ind] == 1.:
            active_site_list.append(site_ind)
    
    if active_site_list == []:
        return 0
    else:
        return np.sum( nn_pred.predict( all_trans[active_site_list,:] ) ) / normalize_fac



def optimize(input):
    
    '''
    Simulated annealing optimization - maximizes the objective function
    
    :param init_struc: Initial structure
    :param nn_class: classification neural network
    :param nn_pred: Prediction neural network
    '''
    
    cat = input[0]
    nn_class = input[1]
    nn_pred = input[2]
    
    total_steps = 100 * len( cat.variable_occs )
    #total_steps = 100          # for debugging
    
    #initial_T = 0.6
    initial_T = 0
    c = 0.5
    
    # Trajectory recording parameters
    n_record = 100
    step_rec = np.zeros(n_record+1)
    OF_rec = np.zeros(n_record+1)
    record_ind = 0
    steps_to_record = np.linspace(0, total_steps,  n_record+1 )
    steps_to_record = steps_to_record.astype(int)
    
    # Evaluate initial structure
    x = cat.variable_occs
    syms = cat.generate_all_translations()
    OF = eval_rate(syms, nn_class, nn_pred, normalize_fac = 1)
    
    # Record data
    step_rec[record_ind] = 0
    OF_rec[record_ind] = OF
    record_ind += 1
    
    CPU_start = time.time()        
    
    for step in xrange( total_steps ):
        
        #Metro_temp = initial_T * (1 - float(step) / total_steps)            # Linear cooling schedule
        Metro_temp = c / np.log(step+2)                                                      # Logarithmic cooling schedule
        
        
        '''
        Random move
        '''
        
        x_new = [i for i in x]
        syms_new = np.copy(syms)
        ind = random.choice(range(len(x_new)))
    
        if x_new[ind] == 1:
            x_new[ind] = 0
        elif x_new[ind] == 0:
            x_new[ind] = 1
        else:
            raise NameError('Invalid occupancy')
    
        d_flip_1, d_flip_2 = cat.var_ind_to_sym_inds(ind)
        for i in range(len(x)):
            d1, d2 = cat.var_ind_to_sym_inds(i)
            ind_to_flip = cat.sym_inds_to_var_ind(d_flip_1 - d1, d_flip_2 - d2)
            syms_new[i,ind_to_flip] = x_new[ind]

        
        '''
        End random move
        '''
        
        OF_new = eval_rate(syms_new, nn_class, nn_pred, normalize_fac = 1)           # Evaluate the new structure and determine whether or not to accept
        
        if OF_new - OF > 0:                    # Downhill move
            accept = True   
        else:                                   # Uphill move
            if Metro_temp > 0:                  # Finite temperature, may accept uphill moves (changed to downhill because we are maximizing)
                accept = np.exp( ( OF_new - OF ) / Metro_temp ) > random.random()
            else:                               # Zero temperature, never accept uphill moves
                accept = False
        
        # Reverse the change if the move is not accepted
        if accept:
            x = x_new
            syms = syms_new
            OF = OF_new
        
        # Record data
        if (step+1) in steps_to_record:
            step_rec[record_ind] = step+1
            OF_rec[record_ind] = OF
            record_ind += 1
        
            
    CPU_end = time.time()
    print('Simulated annealing time elapsed: ' + str(CPU_end - CPU_start) )
    
    cat.assign_occs(x)

    return [ x , np.array([step_rec, OF_rec])]
