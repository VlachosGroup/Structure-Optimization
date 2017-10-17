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

# functions
from read_KMC_site_props import *
from train_neural_nets import *
from optimize_structure import *
from generate_more_structures import *

if __name__ == '__main__':

    initial_DB_size = 98            # could also use 96
    gen_size = 3           # 96 KMC simulations can run at one time
    start_iteration = 1
    end_iteration = 2
    data_fldr = '/home/vlachos/mpnunez/NN_data/AB_data'
    DB_fldr = './KMC_DB'        # relative to data_fldr
    
    # Initialize optimization with random structures
    structure_list = [NiPt_NH3_simple() for i in xrange(gen_size)]
    for struc in structure_list:
        struc.randomize()

    for iteration in range(start_iteration-1, end_iteration):

        os.chdir(data_fldr)
    
        '''
        (1) read KMC data -> build X and Y, get intial structures for optimization
        '''
        
        n_fldrs = initial_DB_size + iteration * gen_size
    
        fldr_list = [os.path.join(DB_fldr, 'structure_' + str(i+1) ) for i in xrange(n_fldrs)]
    
        # Read from existing pickle file
        if iteration == 0 and os.path.isfile('Iteration_1_X.npy'): 
    
            structure_occs = np.load('Iteration_1_X.npy')
            site_propensities = np.load('Iteration_1_Y.npy')
        
        # Read all KMC files
        else:
        
            # Run in parallel
            pool = Pool()
            x_vec = pool.map(read_x, fldr_list)
            pool.close()
            
            pool = Pool()
            y_vec = pool.map(read_y, fldr_list)
            pool.close()
            
            structure_occs = np.array(x_vec)
            site_propensities = np.array(y_vec)
            
            np.save('Iteration_' + str(iteration+1) + '_X.npy', structure_occs)
            np.save('Iteration_' + str(iteration+1) + '_Y.npy', site_propensities)
        
        '''
        (2) X and Y -> neural network objects
        '''
        
        nn_list = train_neural_nets(structure_occs, site_propensities, iter_num = iteration+1)
        nn_class = nn_list[0]
        nn_pred = nn_list[1]
        
        
        '''
        (3) optimize: neural network objects + initial structures -> optimized structures
        '''
        
        #pool = Pool()
        #structure_pop = pool.map(optimize, structure_pop)
        #pool.close()
        trajectory_list = []
        predicted_activities = []
        for struc in structure_list:
            trajectory = optimize(struc, nn_class, nn_pred)
            trajectory_list.append(np.array(trajectory))
            print '\nOptimization complete'
            print 'Predicted activity: ' + str(trajectory[1][-1])
            predicted_activities.append(trajectory[1][-1])
        
        '''
        (4) optimized structures -> KMC input files
        '''
        
        # NEED TO PARALLELIZE / SPEED UP
        
        for new_calc_ind in xrange(gen_size):
            build_KMC_input(structure_list[new_calc_ind], os.path.join(DB_fldr, 'structure_' + str(n_fldrs + new_calc_ind + 1) ),
                    trajectory = trajectory_list[new_calc_ind])
        
        '''
        (5) run KMC simulations
        '''
        
        rep = zw.Replicates()
        rep.ParentFolder = data_fldr
        rep.run_dirs = [os.path.join(DB_fldr, 'structure_' + str(n_fldrs + new_calc_ind + 1) ) for new_calc_ind in xrange(gen_size)]
        rep.n_trajectories = gen_size
        rep.runtemplate = zw.kmc_traj()
        rep.runtemplate.exe_file = '/home/vlachos/mpnunez/bin/zacros_ML.x'
        rep.RunAllTrajectories_JobArray(max_cores = 80, server = 'Squidward', job_name = 'zacros_JA')
        
        # Remove extra files from the job array run
        os.remove('zacros_submit_JA.qs')
        os.remove('dir_list.txt')
        os.system('rm *.po*')
        os.system('rm *.o*')