# Use the neural networks and optimize the structure

import numpy as np
import random
import time
import os

import matplotlib as mat
import matplotlib.pyplot as plt

def optimize(cat, surrogate, syms = None, n_cycles = 100, T_0 = 0.05, cooling = 'exp', fldr = None):
    
    '''
    Simulated annealing optimization - maximizes the objective function
    
    :param cat: Initial structure
    :param surrogate: Surrogate model
    :param syms: All translations of the catalyst structure
    :param n_cycles: Multiple by the number of Ni sites to get the total number of Metropolis steps
    :param c: Cooling schedule parameter. Should be comparable than the largest possible change in the objective function
    '''
    
    total_steps = n_cycles * len( cat.variable_occs )
    tau = total_steps / 5.0
    
    # Trajectory recording parameters
    n_record = 100
    step_rec = np.zeros(n_record+1)
    OF_rec = np.zeros(n_record+1)
    temps = np.zeros(n_record+1)
    record_ind = 0
    steps_to_record = np.linspace(0, total_steps,  n_record+1 )
    steps_to_record = steps_to_record.astype(int)
    
    # Evaluate initial structure
    x = cat.variable_occs
    if syms is None:
        syms = cat.generate_all_translations()
    OF = surrogate.eval_rate( syms )
    
    # Record data
    step_rec[record_ind] = 0
    OF_rec[record_ind] = OF / cat.atoms_per_layer
    temps[record_ind] = T_0
    record_ind += 1
    delta_OFs = np.zeros(total_steps)
    
    CPU_start = time.time()        
    
    
    
    for step in xrange( total_steps ):
        
        if cooling == 'log':            # Logarithmic cooling schedule
            Metro_temp = T_0 / np.log(step+2)                                                      
        elif cooling == 'exp':         # Exponential cooling schedule
            Metro_temp = T_0 * np.exp( - step / tau )
        elif cooling == 'linear':         # Linear cooling schedule
            Metro_temp = T_0 * ( 1 - ( step + 1) / total_steps )
        elif cooling == 'quench':
            Metro_temp = 0
        
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
        
        OF_new = surrogate.eval_rate( syms_new )   # Evaluate the new structure and determine whether or not to accept
        delta_OF = OF_new - OF
        
        if delta_OF > 0:                    # Downhill move
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
            OF_rec[record_ind] = OF / cat.atoms_per_layer
            temps[record_ind] = Metro_temp
            record_ind += 1
        
        delta_OFs[step] = delta_OF
        
            
    CPU_end = time.time()
    print('Simulated annealing time elapsed: ' + str(CPU_end - CPU_start) )
    
    cat.assign_occs(x)
    
    '''
    Plot information from the optimization
    '''
    
    if fldr is None:
        return
    
    trajectory = np.array([step_rec, OF_rec])
    
    # Put a plot of the optimization trajectory in the scaledown folder for each structure
    plt.figure()
    plt.plot(trajectory[0,:], trajectory[1,:], '-')
    plt.xticks(size=18)
    plt.yticks(size=18)
    plt.xlabel('Metropolis step', size=24)
    plt.ylabel('Structure rate', size=24)
    plt.xlim([trajectory[0,0], trajectory[0,-1]])
    plt.ylim([0, None])
    plt.tight_layout()
    plt.savefig(os.path.join(fldr, 'sim_anneal_trajectory'), dpi = 600)
    plt.close()
    
    # Put a plot of the optimization trajectory in the scaledown folder for each structure
    np.save(os.path.join(fldr, 'sim_anneal_trajectory.npy'), trajectory)
    
    plt.figure()
    plt.plot(trajectory[0,:], temps, '-')
    plt.xticks(size=18)
    plt.yticks(size=18)
    plt.xlabel('Metropolis step', size=24)
    plt.ylabel('Temperature', size=24)
    plt.xlim([trajectory[0,0], trajectory[0,-1]])
    plt.ylim([0, None])
    plt.tight_layout()
    plt.savefig(os.path.join(fldr, 'temperature_profile'), dpi = 600)
    plt.close()
    
    #plt.figure()
    #plt.hist(delta_OFs, bins = 50)
    #plt.xlabel(r'\Delta OF', size=24)
    #plt.ylabel('Frequency', size=24)
    ##plt.xlim([trajectory[0,0], trajectory[0,-1]])
    ##plt.ylim([0, 300])
    #plt.xticks(size=20)
    #plt.yticks(size=20)
    #plt.tight_layout()
    #plt.savefig(os.path.join(fldr, 'delta_OF_hist'), dpi = 600)
    #plt.close()
