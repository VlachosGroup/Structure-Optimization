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
from KMC_handler import *
from train_surrogate import surrogate
from optimize_SA import *

from mpi4py import MPI      # MPI parallelization

if __name__ == '__main__':

    '''
    User input
    '''
    
    initial_DB_size = 16
    gen_size = 16                    # 96 KMC simulations can run at one time
    start_iteration = 1
    end_iteration = 10
    data_fldr = '/home/vlachos/mpnunez/OML_data/NH3_data_2/OML_data'
    DB_fldr = '/home/vlachos/mpnunez/OML_data/NH3_data_2/KMC_DB'
    kmc_input_fldr = '/home/vlachos/mpnunez/OML_data/NH3_data_2/KMC_input'
    exe_file = '/home/vlachos/mpnunez/bin/zacros_ML.x'
    rate_rescale = False
    n_kmc_reps = 5
    
    sys.setrecursionlimit(1500)             # Needed for large number of atoms
    cat = NiPt_NH3()
    
    # Run in parallel
    COMM = MPI.COMM_WORLD
    COMM.Barrier()
    
    '''
    Read KMC data from database
    '''
    
    fldr_list = [os.path.join(DB_fldr, 'structure_' + str(i+1) ) for i in xrange(initial_DB_size)]
    sym_list = []
    site_rate_list = []
    for fldr in fldr_list:
        sym_list.append( np.load(os.path.join(fldr, 'occ_symmetries.npy')) )
        site_rate_list.append( np.load(os.path.join(fldr, 'site_rates.npy')) )
    
    structure_occs = np.vstack(sym_list)
    site_rates = np.vstack(site_rate_list)

    
    '''
    Initialize new structures
    '''
    

    # Initialize optimization with random structures
    structure_list = [NiPt_NH3() for i in xrange(gen_size)]
    intial_struc_inds = random.sample(range(initial_DB_size), gen_size)     # choose some of training structures as initial structures
    
    for i in range(gen_size):
        structure_list[i].assign_occs( structure_occs[ 3 * cat.atoms_per_layer * intial_struc_inds[i], :] )
    
    # Scatter list of structures and structure IDs onto different processors
    structure_list_proc = [structure_list[_i::COMM.size] for _i in range(COMM.size)]
    structure_IDs_all = range(initial_DB_size - gen_size+1, initial_DB_size+1)
    structure_IDs = [structure_IDs_all[_i::COMM.size] for _i in range(COMM.size)]
    structure_list_proc = COMM.scatter(structure_list_proc, root=0) 
    structure_IDs = COMM.scatter(structure_IDs, root=0) 
    
    '''
    Online learning loop
    '''
    
    for iteration in range(start_iteration-1, end_iteration):
    
        '''
        Update training data
        '''
        
        if iteration > 0:
            structure_occs = np.vstack([structure_occs, structure_occs_new])
            site_rates = np.concatenate([site_rates, site_rates_new], axis = 0)
        
        structure_rates_KMC = np.sum(site_rates, axis = 1) / cat.atoms_per_layer      # add site rates to get structure rates
        max_site_rate = np.max(site_rates)      # Use maximum site rate to set the cooling schedule in optimization
        
        '''
        Train the surrogate model
        '''
        
        surr = surrogate()
        surr.all_syms = structure_occs
        surr.partition_data_set(site_rates)
        surr.train_classifier()
        surr.train_regressor(reg_parity_fname = 'Iteration_' + str(iteration+1) + '_site_parity')

        raise NameError('stop')
        '''
        Evaluate structures in the training set with the surrogate model
        '''

        n_training_strucs = len(structure_rates_KMC)
        n_sites = site_rates.shape[1]
        structure_rates_NN = np.zeros(n_training_strucs)
        
        for i in xrange(n_training_strucs):
            syms = surr.all_syms[ i * n_sites * 3 : (i+1) * n_sites * 3 : 3 , :]         # extract translations only from the symmetries
            structure_rates_NN[i] = surr.eval_rate( syms ) / cat.atoms_per_layer
        
        if COMM.rank == 0:
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
            plt.legend(['Training (' + str(len(structure_rates_KMC)) + ')', 'Optima'], loc=4, prop={'size':20}, frameon=False)
            plt.tight_layout()
            plt.savefig(os.path.join(data_fldr, 'Iteration_' + str(iteration+1) + '_optimum_parity'), dpi = 600)
            plt.close()
        
        
        '''
        Optimize each structure and evaluate with KMC
        '''
        
        # Update structure IDs
        structure_IDs = [i + gen_size for i in structure_IDs]
        
        for ind in xrange(len(structure_list_proc)):
        
            '''
            Optimize structures using the surrogate model and simulated annealing
            '''
        
            struc_folder = os.path.join(DB_fldr, 'structure_' + str(structure_IDs[ind]) )
            all_syms = np.load(os.path.join(struc_folder, 'occ_symmetries.npy'))
            
            struc = structure_list_proc[ind]
            outputs = optimize( struc, surr, syms = all_syms[::3,:], c = 2*max_site_rate)       # [ x , syms, np.array([step_rec, OF_rec])]
            trajectory = outputs[2]

            '''
            Write structure information into database folder
            '''

            # Write data for structure
            
            write_structure_files(struc, struc_folder, all_symmetries = outputs[1])
            
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
            plt.savefig(os.path.join(struc_folder, 'sim_anneal_trajectory'), dpi = 600)
            plt.close()
            
            # Put a plot of the optimization trajectory in the scaledown folder for each structure
            np.save(os.path.join(struc_folder, 'sim_anneal_trajectory.npy'), trajectory)
            
            '''
            Run KMC and compute site rates
            '''
            
            cum_reps = steady_state_rescale(kmc_input_fldr, struc_folder, exe_file, 'N2', n_runs = 5, n_batches = 1000, 
                                prod_cut = 1000, include_stiff_reduc = True, max_events = int(1e3), 
                                max_iterations = 20, ss_inc = 1.0, n_samples = 100,
                                rate_tol = 0.05)
                                
            site_rates_onestruc = compute_site_rates(struc, cum_reps, gas_prod = 'N2', gas_stoich = 1)
            np.save(os.path.join(struc_folder, 'site_rates.npy'), site_rates_onestruc)
            
        COMM.Barrier()      # Wait for all processors to run their KMC simulations to completion    
    
        '''
        Read new KMC files
        '''
        
        first_fldr = initial_DB_size + iteration * gen_size
        last_fldr = initial_DB_size + (iteration+1) * gen_size
        fldr_list = [os.path.join(DB_fldr, 'structure_' + str(i+1) ) for i in xrange(first_fldr, last_fldr)]
        sym_list = []
        site_rate_list = []
        predicted_activities = []
        for fldr in fldr_list:
            sym_list.append( np.load(os.path.join(fldr, 'occ_symmetries.npy')) )
            site_rate_list.append( np.load(os.path.join(fldr, 'site_rates.npy')) )
            traj = np.load(os.path.join(fldr, 'sim_anneal_trajectory.npy'))
            predicted_activities.append(traj[-1,1])
        
        structure_occs_new = np.vstack(sym_list)
        site_rates_new = np.vstack(site_rate_list)
        
        # Compute KMC evaluated and predicted rates for the new structures
        structure_rates_KMC_new = np.sum(site_rates_new, axis = 1) / cat.atoms_per_layer      # add site rates to get structure rates           
        predicted_activities = np.array(predicted_activities) / cat.atoms_per_layer
        
        '''
        Plot surrogate parity 
        '''
        
        if COMM.rank == 0:
        
            # Save parity plot data for later analysis
            np.save(os.path.join(data_fldr, 'Iteration_' + str(iteration+1) + '_structure_rates_KMC.npy'), structure_rates_KMC)
            np.save(os.path.join(data_fldr, 'Iteration_' + str(iteration+1) + '_structure_rates_NN.npy'), structure_rates_NN)
            np.save(os.path.join(data_fldr, 'Iteration_' + str(iteration+1) + '_structure_rates_KMC_new.npy'), structure_rates_KMC_new)
            np.save(os.path.join(data_fldr, 'Iteration_' + str(iteration+1) + '_predicted_activities.npy'), predicted_activities)
            
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
            plt.legend(['Training (' + str(len(structure_rates_KMC)) + ')', 'Optima'], loc=4, prop={'size':20}, frameon=False)
            plt.tight_layout()
            plt.savefig(os.path.join(data_fldr, 'Iteration_' + str(iteration+1) + '_optimum_parity'), dpi = 600)
            plt.close()