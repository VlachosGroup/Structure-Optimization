'''
Evaluate database structures
'''

import sys
sys.path.append('/home/vlachos/mpnunez/Github/Zacros-Wrapper')
sys.path.append('/home/vlachos/mpnunez/Github/Structure-Optimization')
sys.path.append('/home/vlachos/mpnunez/python_packages/ase')
sys.path.append('/home/vlachos/mpnunez/python_packages/sklearn/lib/python2.7/site-packages')

import os
import shutil
import numpy as np

import random
import time
from multiprocessing import Pool
from mpi4py import MPI      # MPI parallelization

import zacros_wrapper as zw
from OML.NiPt_NH3 import *
from OML.toy_cat import *
from OML.KMC_handler import *
from OML.train_surrogate import *
from OML.optimize_SA import *


if __name__ == '__main__': 
    
    ''' User input '''
    n_strucs = 48
    cat = NiPt_NH3()
    DB_fldr = '/home/vlachos/mpnunez/OML_data/AB_data_5/KMC_DB'
    kmc_input_fldr = '/home/vlachos/mpnunez/OML_data/AB_data_5/KMC_input'
    exe_file = '/home/vlachos/mpnunez/bin/zacros_ML.x'
    ''' '''
    
    dir_list = range(n_strucs)
    
    for dir_ind in range(len(dir_list)):
        dir_list[dir_ind] = os.path.join(DB_fldr, 'structure_' + str(dir_list[dir_ind]+1))
    
    # Run in parallel
    COMM = MPI.COMM_WORLD
    COMM.Barrier()
    
    # Collect whatever has to be done in a list. Here we'll just collect a list of
    # numbers. Only the first rank has to do this.
    if COMM.rank == 0:
        jobs = [dir_list[_i::COMM.size] for _i in range(COMM.size)]             # Split into however many cores are available.
    else:
        jobs = None
    
    jobs = COMM.scatter(jobs, root=0)           # Scatter jobs across cores.
    
    # Now each rank just does its jobs and collects everything in a results list.
    # Make sure to not use super big objects in there as they will be pickled to be
    # exchanged over MPI.
    for job in jobs:
        struc_name = os.path.basename(os.path.normpath(job))
        struc_name = struc_name[-5:]
        cum_reps = steady_state_rescale(kmc_input_fldr, job, exe_file, 'N2', n_runs = 10, n_batches = 1000, 
                                prod_cut = 1500, include_stiff_reduc = False, max_events = int(1e3), 
                                max_iterations = 15, ss_inc = 1.0, n_samples = 100,
                                rate_tol = 0.05, j_name = struc_name)
        
        cum_reps.runAvg.lat.Read_lattice_output( os.path.join(job,'Iteration_1','1') )
        cum_reps.AverageRuns()
        site_rates = compute_site_rates(cat, cum_reps.runAvg, gas_prod = 'N2', gas_stoich = 1)
        np.save(os.path.join(job, 'site_rates.npy'), site_rates)