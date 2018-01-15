'''
Driver for the online machine learning program
'''

import sys
sys.path.append('/home/vlachos/mpnunez/Github/Zacros-Wrapper')
sys.path.append('/home/vlachos/mpnunez/Github/Structure-Optimization')
sys.path.append('/home/vlachos/mpnunez/python_packages/ase')
sys.path.append('/home/vlachos/mpnunez/python_packages/sklearn/lib/python2.7/site-packages')

import os
import pickle
import matplotlib as mat
import matplotlib.pyplot as plt
from multiprocessing import Pool
from mpi4py import MPI      # MPI parallelization

import zacros_wrapper as zw
from OML.NiPt_NH3 import *
from OML.toy_cat import *
from OML.KMC_handler import *
from OML.train_surrogate import *
from OML.optimize_SA import *


def read_database(fldr_list):
    '''
    Read all data in the database from structures that are finished
    :fldr_list: List of folders that possibly have data
    '''  
    
    structures_evaluated = []
    sym_list = []
    site_rate_list = []
    i = 0
    for fldr in fldr_list:
        if os.path.isfile(os.path.join(fldr, 'site_rates.npy')):        # skip incomplete calculations
            structures_evaluated.append(i)
            sym_list.append( np.load(os.path.join(fldr, 'occ_symmetries.npy')) )
            #sym_list.append( np.load(os.path.join(fldr, 'site_types_all_syms.npy')) )
            site_rate_list.append( np.load(os.path.join(fldr, 'site_rates.npy')) )
            i += 1
        
    structure_occs = np.vstack(sym_list)
    site_rates = np.vstack(site_rate_list)
    
    return [structure_occs, site_rates]     # Return list of all symmetries and site rates
    

def plot_parity(curr_fldr, structure_rates_KMC, structure_rates_NN, structure_rates_KMC_new = None, predicted_activities = None):
    '''
    :param curr_fldr: Folder in which to print the graph
    :param Plot surrogate parity: 
    :param structure_rates_KMC: training y data
    :param structure_rates_NN: training y prediction
    :param structure_rates_KMC_new: test y data
    :param predicted_activities: test y prediction
    '''

    mat.rcParams['mathtext.default'] = 'regular'
    mat.rcParams['text.latex.unicode'] = 'False'
    mat.rcParams['legend.numpoints'] = 1
    mat.rcParams['lines.linewidth'] = 2
    mat.rcParams['lines.markersize'] = 12

    plt.figure()
    if structure_rates_KMC_new is None:
        all_point_values = np.hstack([structure_rates_KMC, structure_rates_NN])
    else:
        all_point_values = np.hstack([structure_rates_KMC, structure_rates_NN, structure_rates_KMC_new, predicted_activities])
    par_min = min( all_point_values )
    par_max = max( all_point_values )
    plt.plot( [par_min, par_max], [par_min, par_max], '--', color = 'k', label = None)
    plt.plot(structure_rates_KMC, structure_rates_NN, 'o', label = 'Training (' + str(len(structure_rates_KMC)) + ')')
    if not structure_rates_KMC_new is None:
        plt.plot(structure_rates_KMC_new, predicted_activities, 's', label = 'Optima')
    
    plt.xticks(size=18)
    plt.yticks(size=18)
    plt.xlabel(r'Kinetic Monte Carlo ($r^{KMC}(\sigma)$) ($s^{-1}$)', size=24)
    plt.ylabel(r'Surrogate ($r^{surr}(\sigma)$) ($s^{-1}$)', size=24)
    plt.legend(loc=4, prop={'size':20}, frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(curr_fldr, 'structure_parity'), dpi = 600)
    plt.close()
        
    
if __name__ == '__main__':

    '''
    User input
    '''
    
    # Calculation size
    n_kmc_reps = 5              # Number of replicate trajectories per KMC evaluation
    max_iterations = 10         # Maximum number of iterations for the online machine learning
    
    # Directories
    main_fldr = '/home/vlachos/mpnunez/OML_data/NH3_lumped_2'
    DB_fldr = os.path.join(main_fldr, 'KMC_DB')
    kmc_input_fldr = os.path.join(main_fldr, 'KMC_input')
    exe_file = '/home/vlachos/mpnunez/bin/zacros_ML.x'
    structure_proc = NiPt_NH3()                     # structure for this processor

    # Run in parallel
    COMM = MPI.COMM_WORLD
    COMM.Barrier()
    gen_size = COMM.Get_size()
    
    # Initialize new structures
    sys.setrecursionlimit(1500)             # Needed for large number of atoms
    structure_proc.randomize(coverage = 0.5)        # randomize the occupancies

    
    '''
    Online learning loop
    '''
    
    for iteration in range(1,max_iterations):
    
        '''
        Read all training data from the database
        '''
        
        # Read database
        db_fldr_list = [os.path.join(DB_fldr, o) for o in os.listdir(DB_fldr) if os.path.isdir(os.path.join(DB_fldr,o))]
        db_info = read_database(db_fldr_list)
        structure_occs = db_info[0]
        site_rates = db_info[1]
        
        structure_rates_KMC = np.sum(site_rates, axis = 1) / structure_proc.atoms_per_layer      # add site rates to get structure rates
        max_site_rate = np.max(site_rates)      # Use maximum site rate to set the cooling schedule in optimization
        
        
        '''
        Make folder for the new structure
        '''
        
        # Assign folder name based on processor and iteration
        curr_fldr = os.path.join(DB_fldr, 'structure_' + str(COMM.rank) + '_' + str(iteration) )
        
        # Make folder for new structure
        if not os.path.exists(curr_fldr ):
            os.makedirs(curr_fldr )
    
        # If this folder has already been created and evaluated, skip to the next iteration...

        # Clear folder contents
        for the_file in os.listdir(curr_fldr ):
            file_path = os.path.join(curr_fldr , the_file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(e)
        
        '''
        Train the surrogate model
        '''        
        
        # This needs to be cleaned up. It doesn't make sense to put X as a class variable, but not Y...
        surr = surrogate()
        surr.all_syms = structure_occs      
        surr.partition_data_set(site_rates, structure_proc)
        #surr.train_decision_tree_regressor(site_rates)
        #surr.partition_data_set(site_rates)
        surr.train_classifier()
        surr.train_regressor(reg_parity_fname = os.path.join( curr_fldr , 'site_parity'))

        '''
        Evaluate structures in the training set with the surrogate model
        '''

        n_training_strucs = len(structure_rates_KMC)
        n_sites = site_rates.shape[1]
        structure_rates_NN = np.zeros(n_training_strucs)
        
        for i in xrange(n_training_strucs):
            syms = surr.all_syms[ i * n_sites * 3 : (i+1) * n_sites * 3 : 3 , :]         # extract translations only from the symmetries
            structure_rates_NN[i] = surr.eval_rate( syms ) / structure_proc.atoms_per_layer
        
        plot_parity(curr_fldr, structure_rates_KMC, structure_rates_NN)
        

        '''
        Optimize the structure using the surrogate model and simulated annealing
        '''
        
        all_syms = structure_proc.generate_all_translations_and_rotations()
        optimize( structure_proc, surr, syms = all_syms[::3,:], n_cycles = 500, T_0 = max_site_rate, fldr = curr_fldr)
        
        
        '''
        Write structure information into database folder
        '''

        structure_proc.graph_to_KMClattice()        # build KMC lattice
        write_structure_files(structure_proc, curr_fldr, all_symmetries = None)

        '''
        Run KMC and compute site rates
        '''
        
        cum_reps = steady_state_rescale(kmc_input_fldr, curr_fldr, exe_file, 'N2', n_runs = 5, n_batches = 1000, 
                            prod_cut = 500, include_stiff_reduc = True, max_events = int(1e3), 
                            max_iterations = 10, ss_inc = 1.0, n_samples = 100,
                            rate_tol = 0.05, j_name = 'struc_' + str(iteration) + '_' + str(COMM.rank) )
        
        cum_reps.runAvg.lat.Read_lattice_output( os.path.join(curr_fldr,'Iteration_1','1') )            
        cum_reps.AverageRuns()
        site_rates_onestruc = compute_site_rates(structure_proc, cum_reps.runAvg, gas_prod = 'N2', gas_stoich = 1)
        np.save(os.path.join(curr_fldr, 'site_rates.npy'), site_rates_onestruc)
                
    
        '''
        Read new KMC files
        '''
        
        sym_list = []
        site_rate_list = []
        predicted_activities = []
        for fldr in [curr_fldr]:
            sym_list.append( np.load(os.path.join(fldr, 'occ_symmetries.npy')) )
            site_rate_list.append( np.load(os.path.join(fldr, 'site_rates.npy')) )
            traj = np.load(os.path.join(fldr, 'sim_anneal_trajectory.npy'))
            predicted_activities.append(traj[1,-1])
        
        site_rates_new = np.vstack(site_rate_list)
        
        # Compute KMC evaluated and predicted rates for the new structures
        structure_rates_KMC_new = np.sum(site_rates_new, axis = 1) / structure_proc.atoms_per_layer      # add site rates to get structure rates           
        predicted_activities = np.array(predicted_activities)
        
        # Save parity plot data for later analysis
        np.save(os.path.join(curr_fldr, 'structure_rates_KMC.npy'), structure_rates_KMC)
        np.save(os.path.join(curr_fldr, 'structure_rates_NN.npy'), structure_rates_NN)
        plot_parity(curr_fldr, structure_rates_KMC, structure_rates_NN, structure_rates_KMC_new = structure_rates_KMC_new, predicted_activities = predicted_activities)
        
        # Terminate if the KMC is the best it has ever been and was accurately predicted by the surrogate