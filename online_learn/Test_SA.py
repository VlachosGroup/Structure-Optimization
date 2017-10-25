'''
Driver for the online learning program
'''

import sys
sys.path.append('/home/vlachos/mpnunez/Github/Zacros-Wrapper')
sys.path.append('/home/vlachos/mpnunez/Github/Structure-Optimization')
sys.path.append('/home/vlachos/mpnunez/python_packages/ase')
sys.path.append('/home/vlachos/mpnunez/python_packages/sklearn/lib/python2.7/site-packages')

import pickle

from multiprocessing import Pool
from NH3.NiPt_NH3_simple import NiPt_NH3_simple
import zacros_wrapper as zw

import matplotlib as mat
import matplotlib.pyplot as plt

# functions
from read_KMC_site_props import *
from train_neural_nets import *
from optimize_structure import *
from generate_more_structures import *

if __name__ == '__main__':

    initial_DB_size = 113
    gen_size = 1                    # 96 KMC simulations can run at one time
    start_iteration = 1
    end_iteration = 10
    data_fldr = '/home/vlachos/mpnunez/NN_data/AB_data'
    DB_fldr = './KMC_DB'        # relative to data_fldr
    
    os.chdir(data_fldr)
    
    '''
    Read KMC data -> build X and Y
    '''
    
    # Read from existing pickle file
    structure_occs = np.load('Iteration_0_X.npy')
    
    structure_occs_new = None
    site_propensities_new = None

    
    # Initialize optimization with random structures
    structure_list = [NiPt_NH3_simple() for i in xrange(gen_size)]
    intial_struc_inds = random.sample(range(initial_DB_size), gen_size)     # choose some of training structures as initial structures
    
    for i in range(gen_size):
        structure_list[i].assign_occs( structure_occs[ intial_struc_inds[i], :] )
    
    #for struc in structure_list:
    #    struc.randomize()           # Use an initial random structure or a training structure
    
    '''
    Online learning loop
    '''
    
    iteration =1

    os.chdir(data_fldr)
    
    '''
    X and Y -> neural network objects
    '''
    
    # If this isn't the first iteration, add data from new structures

    
    nn_class = pickle.load( open( "Iteration_1_nn_class.p", "rb" ) )
    nn_pred = pickle.load( open( "Iteration_1_nn_reg.p", "rb" ) )


    '''
    Optimize: neural network objects + initial structures -> optimized structures
    '''
    
    input_list = [ [ struc, nn_class, nn_pred ] for struc in structure_list ]
    
    #trajectory_list = pool.map(optimize, input_list)
    outputs = optimize( input_list[0] )
    

    structure_list[i].assign_occs(outputs[0])
    traj = outputs[1]
        
    
    
    # Serial version
    #trajectory_list = []
    #predicted_activities = []
    #for struc in structure_list:
    #    trajectory = optimize(struc, nn_class, nn_pred)
    #    trajectory_list.append(np.array(trajectory))
    #    predicted_activities.append(trajectory[1][-1])