'''
Create random structures for the initial database
'''

import sys
sys.path.append('/home/vlachos/mpnunez/Github/Zacros-Wrapper')
sys.path.append('/home/vlachos/mpnunez/Github/Structure-Optimization/')
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

def make_random_structure(i, n_strucs, cat, fldr_name):

    '''
    Create a random island structure
    :param i: Index of this structure
    :param n_strucs: Total number of structure being made
    :param cat: Catalyst structure
    :param fldr_name: Folder to write the files into
    '''
    
    '''
    Make islands
    '''
    
    np.random.seed(i)
    cat.variable_occs = [0 for j in range(cat.atoms_per_layer)]
    n_tops = int( (3 + 15 * float(i) / (n_strucs+1)))
    #n_tops = 0 # testing
    top_sites = np.random.choice( range(len(cat.variable_atoms)), size=n_tops, replace=False )
    
    for i in top_sites:     # Pick a site to build an island around
        
        sym_inds = cat.var_ind_to_sym_inds(i)
        
        # Small island
        #ind1 = cat.sym_inds_to_var_ind(sym_inds[0], sym_inds[1])
        #cat.variable_occs[ind1] = 1
        #ind1 = cat.sym_inds_to_var_ind(sym_inds[0] - 1, sym_inds[1] + 1)
        #cat.variable_occs[ind1] = 1
        #ind1 = cat.sym_inds_to_var_ind(sym_inds[0], sym_inds[1] + 1)
        #cat.variable_occs[ind1] = 1
        #ind1 = cat.sym_inds_to_var_ind(sym_inds[0] - 1, sym_inds[1])
        #cat.variable_occs[ind1] = 1
        #ind1 = cat.sym_inds_to_var_ind(sym_inds[0] + 1, sym_inds[1])
        #cat.variable_occs[ind1] = 1
        #ind1 = cat.sym_inds_to_var_ind(sym_inds[0], sym_inds[1] - 1)
        #cat.variable_occs[ind1] = 1
        #ind1 = cat.sym_inds_to_var_ind(sym_inds[0] + 1, sym_inds[1] - 1)
        #cat.variable_occs[ind1] = 1
        
        # Big island
        ind1 = cat.sym_inds_to_var_ind(sym_inds[0] - 2, sym_inds[1] + 2)
        cat.variable_occs[ind1] = 1
        ind1 = cat.sym_inds_to_var_ind(sym_inds[0] - 1, sym_inds[1] + 2)
        cat.variable_occs[ind1] = 1
        ind1 = cat.sym_inds_to_var_ind(sym_inds[0] , sym_inds[1] + 2)
        cat.variable_occs[ind1] = 1
        
        ind1 = cat.sym_inds_to_var_ind(sym_inds[0] - 2, sym_inds[1] + 1)
        cat.variable_occs[ind1] = 1
        ind1 = cat.sym_inds_to_var_ind(sym_inds[0] - 1, sym_inds[1] + 1)
        cat.variable_occs[ind1] = 1
        ind1 = cat.sym_inds_to_var_ind(sym_inds[0] , sym_inds[1] + 1)
        cat.variable_occs[ind1] = 1
        ind1 = cat.sym_inds_to_var_ind(sym_inds[0] +1, sym_inds[1] + 1)
        cat.variable_occs[ind1] = 1
        
        ind1 = cat.sym_inds_to_var_ind(sym_inds[0] - 2, sym_inds[1])
        cat.variable_occs[ind1] = 1
        ind1 = cat.sym_inds_to_var_ind(sym_inds[0] - 1, sym_inds[1])
        cat.variable_occs[ind1] = 1
        ind1 = cat.sym_inds_to_var_ind(sym_inds[0] , sym_inds[1] )
        cat.variable_occs[ind1] = 1
        ind1 = cat.sym_inds_to_var_ind(sym_inds[0] +1, sym_inds[1])
        cat.variable_occs[ind1] = 1
        ind1 = cat.sym_inds_to_var_ind(sym_inds[0] +2, sym_inds[1])
        cat.variable_occs[ind1] = 1
        
        ind1 = cat.sym_inds_to_var_ind(sym_inds[0] - 1, sym_inds[1] - 1)
        cat.variable_occs[ind1] = 1
        ind1 = cat.sym_inds_to_var_ind(sym_inds[0] , sym_inds[1] - 1)
        cat.variable_occs[ind1] = 1
        ind1 = cat.sym_inds_to_var_ind(sym_inds[0] +1, sym_inds[1] - 1)
        cat.variable_occs[ind1] = 1
        ind1 = cat.sym_inds_to_var_ind(sym_inds[0] +2, sym_inds[1] - 1)
        cat.variable_occs[ind1] = 1
        
        ind1 = cat.sym_inds_to_var_ind(sym_inds[0] , sym_inds[1] - 2)
        cat.variable_occs[ind1] = 1
        ind1 = cat.sym_inds_to_var_ind(sym_inds[0] + 1, sym_inds[1] - 2)
        cat.variable_occs[ind1] = 1
        ind1 = cat.sym_inds_to_var_ind(sym_inds[0] +2, sym_inds[1] - 2)
        cat.variable_occs[ind1] = 1
    
    n_mutate = 16
    mutate_sites = np.random.choice( range(len(cat.variable_atoms)), size=n_mutate, replace=False )
    for site in mutate_sites:
        if cat.variable_occs[site] == 1:
            cat.variable_occs[site] = 0
        elif cat.variable_occs[site] == 0:
            cat.variable_occs[site] = 1
    
    # Build catalyst structure
    cat.occs_to_atoms()
    cat.occs_to_graph()
    cat.graph_to_KMClattice()
    
    # Build input files
    write_structure_files(cat, fldr_name)
    
    
def main():
    
    ''' User input '''
    n_strucs = 48
    cat = toy_cat()
    DB_fldr = '/home/vlachos/mpnunez/OML_data/AB_data_5/KMC_DB'
    ''' '''
    
    # Run in parallel
    COMM = MPI.COMM_WORLD
    
    # Clear folder contents
    if COMM.rank == 0:
        for the_file in os.listdir(DB_fldr):
            file_path = os.path.join(DB_fldr, the_file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(e)
                
    COMM.Barrier()
    
    # Collect whatever has to be done in a list. Here we'll just collect a list of
    # numbers. Only the first rank has to do this.
    if COMM.rank == 0:
        jobs = range(n_strucs)
        jobs = [jobs[_i::COMM.size] for _i in range(COMM.size)]             # Split into however many cores are available.
    else:
        jobs = None
    
    jobs = COMM.scatter(jobs, root=0)           # Scatter jobs across cores.
    
    # Now each rank just does its jobs and collects everything in a results list.
    # Make sure to not use super big objects in there as they will be pickled to be
    # exchanged over MPI.
    for job in jobs:
        fldr_name = os.path.join(DB_fldr, 'structure_' + str(job+1) )
        make_random_structure(job, n_strucs, cat, fldr_name)
		
if __name__ == '__main__': 
	main()