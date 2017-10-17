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

    initial_DB_size = 96
    gen_size = 3                    # 96 KMC simulations can run at one time
    start_iteration = 1
    end_iteration = 2
    data_fldr = '/home/vlachos/mpnunez/NN_data/AB_data'
    DB_fldr = './KMC_DB'        # relative to data_fldr
    
    os.chdir(data_fldr)
    
    # Initialize optimization with random structures
    structure_list = [NiPt_NH3_simple() for i in xrange(gen_size)]
    for struc in structure_list:
        struc.randomize()
    
    
    '''
    Read KMC data -> build X and Y
    '''
    
    # Read from existing pickle file
    #if os.path.isfile('Iteration_0_X.npy'): 
    if False: 
    
        structure_occs = np.load('Iteration_0_X.npy')
        site_propensities = np.load('Iteration_0_Y.npy')
    
    # Read all KMC files
    else:
    
        fldr_list = [os.path.join(DB_fldr, 'structure_' + str(i+1) ) for i in xrange(initial_DB_size)]
    
        
        for dir in fldr_list:
            read_x(dir)
            read_y(dir)
    
        ## Run in parallel
        #pool = Pool()
        #x_vec = pool.map(read_x, fldr_list)
        #pool.close()
        #
        #pool = Pool()
        #y_vec = pool.map(read_y, fldr_list)
        #pool.close()
        #
        #structure_occs = np.array(x_vec)
        #site_propensities = np.array(y_vec)
        
        #np.save('Iteration_0_X.npy', structure_occs)
        #np.save('Iteration_0_Y.npy', site_propensities)
    
    structure_occs_new = None
    site_propensities_new = None
    raise NameError('stop')
    
    for iteration in range(start_iteration-1, end_iteration):

        os.chdir(data_fldr)
    
        '''
        X and Y -> neural network objects
        '''
        
        # If this isn't the first iteration, add data from new structures
        if iteration > 0:
            structure_occs = np.vstack([structure_occs, structure_occs_new])
            site_propensities = np.stack([site_propensities, site_propensities_new], axis = 0)
        
        nn_list = train_neural_nets(structure_occs, site_propensities, iter_num = iteration+1)
        nn_class = nn_list[0]
        nn_pred = nn_list[1]
        
        
        '''
        Optimize: neural network objects + initial structures -> optimized structures
        '''
        
        # NEED TO PARALLELIZE / SPEED UP
        
        #pool = Pool()
        #structure_pop = pool.map(optimize, structure_pop)
        #pool.close()
        trajectory_list = []
        predicted_activities = []
        for struc in structure_list:
            trajectory = optimize(struc, nn_class, nn_pred)
            trajectory_list.append(np.array(trajectory))
            predicted_activities.append(trajectory[1][-1])
        
        predicted_activities = np.array(predicted_activities)
        
        '''
        Optimized structures -> KMC input files
        '''
        
        n_fldrs = initial_DB_size + iteration * gen_size
        
        for new_calc_ind in xrange(gen_size):
            build_KMC_input(structure_list[new_calc_ind], os.path.join(DB_fldr, 'structure_' + str(n_fldrs + new_calc_ind + 1) ),
                    trajectory = trajectory_list[new_calc_ind])
        
        '''
        Run KMC simulations
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
        
        
        '''
        Read new KMC simulations and see how they performed
        '''
    
        
        x_vec = []
        y_vec = []
        
        for dir in rep.run_dirs:
            x_vec.append( read_x(dir) )
            y_vec.append( read_y(dir) )
        
        structure_occs_new = np.array(x_vec)
        site_propensities_new = np.array(y_vec)
        
        site_rates_new = site_propensities_new[:,:,0] - site_propensities_new[:,:,1]
        structure_rates_KMC_new = np.sum(site_rates_new, axis = 1)
        
        
        np.save('Iteration_' + str(iteration+1) + 'opt_X.npy', structure_occs_new)
        np.save('Iteration_' + str(iteration+1) + 'opt_Y.npy', site_propensities_new)
        
        plt.figure()
        plt.plot(structure_rates_KMC_new, predicted_activities, 'o')
        all_point_values = np.hstack([structure_rates_KMC_new, predicted_activities])
        par_min = min( all_point_values )
        par_max = max( all_point_values )
        plt.plot( [par_min, par_max], [par_min, par_max], '-', color = 'k')  # Can do this for all outputs
        
        plt.xticks(size=18)
        plt.yticks(size=18)
        plt.xlabel('Kinetic Monte Carlo', size=24)
        plt.ylabel('Neural network', size=24)
        #plt.xlim([0, 0.50])
        #plt.ylim([0, 0.50])
        #plt.legend(['train', 'test'], loc=4, prop={'size':20}, frameon=False)
        plt.tight_layout()
        plt.savefig('Iteration_' + str(iteration+1) + '_optimum_parity', dpi = 600)
        plt.close()