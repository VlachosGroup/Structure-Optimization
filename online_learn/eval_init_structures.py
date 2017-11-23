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

from NH3.NiPt_NH3 import NiPt_NH3
from KMC_handler import *

from mpi4py import MPI      # MPI parallelization


if __name__ == '__main__': 
    
    ''' User input '''
    n_strucs = 16
    cat = NiPt_NH3()
    DB_fldr = '/home/vlachos/mpnunez/OML_data/NH3_data_2/KMC_DB'
    kmc_input_fldr = '/home/vlachos/mpnunez/OML_data/NH3_data_2/KMC_input'
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
        cum_reps = steady_state_rescale(kmc_input_fldr, job, exe_file, 'N2', n_runs = 5, n_batches = 1000, 
                                prod_cut = 1000, include_stiff_reduc = True, max_events = int(1e3), 
                                max_iterations = 20, ss_inc = 1.0, n_samples = 100,
                                rate_tol = 0.05)
                                
        site_rates = compute_site_rates(cat, cum_reps, gas_prod = 'N2', gas_stoich = 1)
        np.save(os.path.join(job, 'site_rates.npy'), site_rates)