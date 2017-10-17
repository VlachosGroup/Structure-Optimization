# Use the neural networks and optimize the structure

import numpy as np
import random
import time
import os



'''
Objective function
'''

def eval_rate(cat, sigma, nn_class, nn_pred):

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
        return np.sum( nn_pred.predict( all_trans[active_site_list,:] ) )


'''
Random move
'''    
def rand_move(sigma):

    sigma_new = [i for i in sigma]
    ind = random.choice(range(len(sigma_new)))

    if sigma_new[ind] == 1:
        sigma_new[ind] = 0
    elif sigma_new[ind] == 0:
        sigma_new[ind] = 1
    else:
        raise NameError('Invalid occupancy')

    return sigma_new


def optimize(cat, nn_class, nn_pred):
    
    '''
    Simulated annealing optimization - maximizes the objective function
    
    :param init_struc: Initial structure
    :param nn_class: classification neural network
    :param nn_pred: Prediction neural network
    '''
    
    total_steps = 100
    
    #initial_T = 0.6
    initial_T = 0
    c = 1.
    
    # Trajectory recording parameters
    n_record = 100
    step_rec = np.zeros(n_record+1)
    OF_rec = np.zeros(n_record+1)
    record_ind = 0
    
    
    # Evaluate initial structure
    x = cat.variable_occs
    OF = eval_rate(cat, x, nn_class, nn_pred)
    
    # Record data
    step_rec[record_ind] = 0
    OF_rec[record_ind] = OF
    record_ind += 1
    
    CPU_start = time.time()        
    
    for step in xrange( total_steps ):
        
        #Metro_temp = initial_T * (1 - float(step) / total_steps)            # Linear cooling schedule
        Metro_temp = c / np.log(step+2)                                                      # Logarithmic cooling schedule
        x_new = rand_move(x)                                                # Do a Metropolis move
        OF_new = eval_rate(cat, x_new, nn_class, nn_pred)           # Evaluate the new structure and determine whether or not to accept
        
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
            OF = OF_new
        
        # Record data
        if (step+1) % (total_steps / n_record) == 0:
            step_rec[record_ind] = step+1
            OF_rec[record_ind] = OF
            record_ind += 1
        
            
    CPU_end = time.time()
    #print('Time elapsed: ' + str(CPU_end - CPU_start) )
    
    cat.assign_occs(x)

    return [step_rec, OF_rec]
