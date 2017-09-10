'''
General implementation of multi-objective simulated annealing
See June 22 "Implemented optimization" commit
'''

import numpy as np
import random
import time

def optimize(eval_obj, total_steps = 1000, initial_T = 0.7, n_record = 100):
    
    '''
    Use simulated annealing to optimize defects on the surface
    :param eval_obj: Object to be optimized. Must have get_OF(), rand_move(), and revert_last() methods.
    :param total_steps: Number of Metropolis steps to run
    :param initial_T: Initial temperature (dimensionless). A linear cooling schedule is used
    :param n_record: number of steps to print out (not counting initial state)
    '''
    
    # Evaluate initial structure
    OF = eval_obj.get_OF()     
    print  str(0) + ' steps elapsed:       ' + str(OF)
    
    CPU_start = time.time()        
    
    for step in range( total_steps ):
        
        Metro_temp = initial_T * (1 - float(step) / total_steps)            # Set temperature
        OF_prev = OF                                                        # Record data before changing structure
        eval_obj.rand_move()                                                # Do a Metropolis move
        OF = eval_obj.get_OF()                  # Evaluate the new structure and determine whether or not to accept
        
        if OF - OF_prev < 0:                # Downhill move
            accept = True
        else:                               # Uphill move
            if Metro_temp > 0:              # Finite temperature, may accept uphill moves
                accept = np.exp( - ( OF - OF_prev ) / Metro_temp ) > random.random()
            else:                           # Zero temperature, never accept uphill moves
                accept = False
        
        # Reverse the change if the move is not accepted
        if not accept:
            eval_obj.revert_last()
            OF = OF_prev            # Use previous values for evaluations
        
        # Record data
        if (step+1) % (total_steps / n_record) == 0:
            print  str(step+1) + ' steps elapsed:       ' + str(OF)
        
            
    CPU_end = time.time()
    print('Time elapsed: ' + str(CPU_end - CPU_start) )