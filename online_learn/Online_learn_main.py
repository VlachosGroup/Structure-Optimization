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
from NH3.NiPt_NH3 import NiPt_NH3
from NH3.NiPt_NH3_simple import NiPt_NH3_simple
import zacros_wrapper as zw

import matplotlib as mat
import matplotlib.pyplot as plt

# functions
from read_KMC_site_props import *
from train_surrogate import surrogate
from optimize_SA import *
from build_KMC_input import *

if __name__ == '__main__':

    '''
    User input
    '''
    
    initial_DB_size = 50
    gen_size = 16                    # 96 KMC simulations can run at one time
    start_iteration = 1
    end_iteration = 10
    data_fldr = '/home/vlachos/mpnunez/NN_data/NH3_data_1/OML_data'
    DB_fldr = '/home/vlachos/mpnunez/NN_data/NH3_data_1/KMC_DB'
    kmc_src = '/home/vlachos/mpnunez/NN_data/NH3_data_1/KMC_input'
    
    sys.setrecursionlimit(1500)             # Needed for large number of atoms
    cat = NiPt_NH3()
    
    '''
    Read KMC data from database
    '''
    
    os.chdir(data_fldr)

    # Read from existing pickle file
    if False:#os.path.isfile('Iteration_0_X.npy'): 
    
        structure_occs = np.load('Iteration_0_X.npy')
        site_rates = np.load('Iteration_0_Y.npy')
    
    # Read all KMC files
    else:
    
        fldr_list = [os.path.join(DB_fldr, 'structure_' + str(i+1) ) for i in xrange(initial_DB_size)]
    
        # Read folders in parallel
        output = read_many_calcs(fldr_list)
        structure_occs = output[0]
        site_rates = output[1]
        
        #np.save('Iteration_0_X.npy', structure_occs)
        #np.save('Iteration_0_Y.npy', site_rates)

    structure_occs_new = structure_occs
    
    '''
    Initialize new structures
    '''
    
    surr = surrogate()
    
    # Initialize optimization with random structures
    structure_list = [NiPt_NH3() for i in xrange(gen_size)]
    intial_struc_inds = random.sample(range(initial_DB_size), gen_size)     # choose some of training structures as initial structures
    
    for i in range(gen_size):
        structure_list[i].assign_occs( structure_occs[ intial_struc_inds[i], :] )
    
    #for struc in structure_list:
    #    struc.randomize()           # Use an initial random structure or a training structure
    
    '''
    Online learning loop
    '''
    
    for iteration in range(start_iteration-1, end_iteration):

        os.chdir(data_fldr)
    
        '''
        Update training data
        '''
        
        if iteration > 0:
            structure_occs = np.vstack([structure_occs, structure_occs_new])
            site_rates = np.concatenate([site_rates, site_rates_new], axis = 0)
        
        structure_rates_KMC = np.sum(site_rates, axis = 1) / cat.atoms_per_layer      # add site rates to get structure rates
        
        '''
        Train the surrogate model
        '''
        
        surr.generate_symmetries(structure_occs_new, cat)      # Add symmetries to the list in the surrogate model
        surr.partition_data_set(site_rates)
        surr.train_classifier()
        surr.train_regressor()

        
        '''
        Evaluate structures in the training set with the surrogate model
        '''

        n_training_strucs = len(structure_rates_KMC)
        n_sites = site_rates.shape[1]
        structure_rates_NN = np.zeros(n_training_strucs)
        
        for i in xrange(n_training_strucs):
            syms = surr.all_syms[ i * n_sites * 3 : (i+1) * n_sites * 3 : 3 , :]         # extract translations only from the symmetries
            structure_rates_NN[i] = surr.eval_rate( syms ) / cat.atoms_per_layer
        
        
        plt.figure()
        plt.plot(structure_rates_KMC, structure_rates_NN, 'o')
        all_point_values = np.hstack([structure_rates_KMC, structure_rates_NN ])
        par_min = min( all_point_values )
        par_max = max( all_point_values )
        plt.plot( [par_min, par_max], [par_min, par_max], '-', color = 'k')  # Can do this for all outputs
        
        plt.xticks(size=18)
        plt.yticks(size=18)
        plt.xlabel(r'Kinetic Monte Carlo ($r^{KMC}(\sigma)$) ($s^{-1}$)', size=24)
        plt.ylabel(r'Surrogate ($r^{surr}(\sigma)$) ($s^{-1}$)', size=24)
        #plt.xlim([0, 0.50])
        #plt.ylim([0, 0.50])
        plt.legend(['Training (' + str(len(structure_rates_KMC)) + ')', 'Optima'], loc=4, prop={'size':20}, frameon=False)
        plt.tight_layout()
        plt.savefig('Iteration_' + str(iteration+1) + '_optimum_parity', dpi = 600)
        plt.close()
        
        
        '''
        Optimize structures using the surrogate model -> generate structures to add
        '''
        
        input_list = [ [ struc, surr ] for struc in structure_list ]
        
        pool = Pool()
        outputs = pool.map(optimize, input_list)
        pool.close()
        
        trajectory_list = []
        predicted_activities = []
        for i in range(len(outputs)):
            structure_list[i].assign_occs(outputs[i][0])
            traj = outputs[i][1]
            trajectory_list.append(traj)
            predicted_activities.append(traj[1][-1])
        
        predicted_activities = np.array(predicted_activities) / cat.atoms_per_layer
        
        '''
        Optimized structures -> KMC input files
        '''
        
        n_fldrs = initial_DB_size + iteration * gen_size
        
        for new_calc_ind in xrange(gen_size):
            build_KMC_input(structure_list[new_calc_ind], os.path.join(DB_fldr, 'structure_' + str(n_fldrs + new_calc_ind + 1) ),
                    kmc_src, trajectory = trajectory_list[new_calc_ind])

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
    
        
        # Read folders in parallel
        output = read_many_calcs(rep.run_dirs)
        structure_occs_new = output[0]
        site_rates_new = output[1]
        
        np.save('Iteration_' + str(iteration+1) + 'opt_X.npy', structure_occs_new)
        np.save('Iteration_' + str(iteration+1) + 'opt_Y.npy', site_rates_new)
        
        structure_rates_KMC_new = np.sum(site_rates_new, axis = 1) / cat.atoms_per_layer      # add site rates to get structure rates
        
        '''
        Plot surrogate parity 
        '''
        
        # Save parity plot data for later analysis
        np.save('Iteration_' + str(iteration+1) + '_structure_rates_KMC.npy', structure_rates_KMC)
        np.save('Iteration_' + str(iteration+1) + '_structure_rates_NN.npy', structure_rates_NN)
        np.save('Iteration_' + str(iteration+1) + '_structure_rates_KMC_new.npy', structure_rates_KMC_new)
        np.save('Iteration_' + str(iteration+1) + '_predicted_activities.npy', predicted_activities)
        
        plt.figure()
        plt.plot(structure_rates_KMC, structure_rates_NN, 'o')
        plt.plot(structure_rates_KMC_new, predicted_activities, 's')
        all_point_values = np.hstack([structure_rates_KMC, structure_rates_NN, structure_rates_KMC_new, predicted_activities])
        par_min = min( all_point_values )
        par_max = max( all_point_values )
        plt.plot( [par_min, par_max], [par_min, par_max], '-', color = 'k')  # Can do this for all outputs
        
        plt.xticks(size=18)
        plt.yticks(size=18)
        plt.xlabel(r'Kinetic Monte Carlo ($r^{KMC}(\sigma)$) ($s^{-1}$)', size=24)
        plt.ylabel(r'Surrogate ($r^{surr}(\sigma)$) ($s^{-1}$)', size=24)
        #plt.xlim([0, 0.50])
        #plt.ylim([0, 0.50])
        plt.legend(['Training (' + str(len(structure_rates_KMC)) + ')', 'Optima'], loc=4, prop={'size':20}, frameon=False)
        plt.tight_layout()
        plt.savefig('Iteration_' + str(iteration+1) + '_optimum_parity', dpi = 600)
        plt.close()