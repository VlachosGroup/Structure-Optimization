import numpy as np
import os
import copy

import zacros_wrapper as zw
from zacros_wrapper.utils import *
from zacros_wrapper.Replicates import *

import matplotlib as mat
import matplotlib.pyplot as plt
    

def write_structure_files(cat, run_folder, all_symmetries = None):
    '''
    Write data so that the structure can be identified by KMC evaluator
    :param cat: Catalyst structure object
    :param run_folder: Folder to write these files into
    '''
    
    if not os.path.exists(run_folder):
        os.makedirs(run_folder)
    
    # Occupancy vector
    np.save(os.path.join(run_folder, 'occupancies.npy'), cat.variable_occs)
    
    # Occupancy vector symmetries
    if all_symmetries is None:
        all_symmetries = cat.generate_all_translations_and_rotations()
    np.save(os.path.join(run_folder, 'occ_symmetries.npy'), all_symmetries)
    
    # Molecular pictures of catalyst structure
    cat.show(fname = os.path.join(run_folder,'structure'), fmat = 'png')
    cat.show(fname = os.path.join(run_folder,'structure'), fmat = 'xsd')
    
    # Picture of KMC lattice
    kmc_lat = cat.KMC_lat.PlotLattice()
    kmc_lat.savefig(os.path.join(run_folder,'lattice.png'),format='png', dpi=600)
    kmc_lat.close()
    
    # KMC lattice input file
    cat.KMC_lat.Write_lattice_input(run_folder)

    
def steady_state_rescale(kmc_template, scale_parent_fldr, n_runs = 16, n_batches = 1000, 
                                prod_cut = 1000, include_stiff_reduc = True, max_events = int(1e3), 
                                max_iterations = 20, ss_inc = 1.0, n_samples = 100,
                                rate_tol = 0.05):

    '''
    Handles rate rescaling and continuation of KMC runs
    
    :param kmc_template:            kmc_traj object with information about the physical system
    :param scale_parent_fldr:       Working directory
    :param n_runs:                  Number of trajectories to run, also the number of processors
    :param include_stiff_reduc:     True to allow for scaledown, False to turn this feature off
    :param max_events:              Maximum number of events for the first iteration
    :param max_iterations:          Maximum number of iterations to use
    :param ss_inc:                  Factor to scale the final time by if you have not yet reached steady state
    :param n_samples:               Number of time points to sample for each trajectory
    ''' 
    
    os.system('rm -r ' + os.path.join(scale_parent_fldr, 'Iteration_*'))        # Delete any existing iteration folders
    sum_file = open(os.path.join(scale_parent_fldr, 'Scaledown_summary.txt'),'w') 
    
    prev_batch = zw.Replicates()       # Set this if the starting iteration is not 1
    initial_states = None
    
    
    SDF_vec = None        # scaledown factors for each iteration
    
    # Convergence variables
    is_steady_state = False
    iteration = 1              
    
    scale_final_time = ss_inc

    while not is_steady_state and iteration <= max_iterations:
        
        # Make folder for iteration
        iter_fldr = os.path.join(scale_parent_fldr, 'Iteration_' + str(iteration))
        if not os.path.exists(iter_fldr):
            os.makedirs(iter_fldr)
            
        # Create object for batch
        cur_batch = zw.Replicates()
        cur_batch.ParentFolder = iter_fldr
        cur_batch.n_trajectories = n_runs
        cur_batch.N_batches = n_batches
        cur_batch.Set_kmc_template(kmc_template)        # Set template KMC trajectory
        
        if iteration == 1:              # Sample on events, because we do not know the time scales
        
            # Set sampling parameters
            cur_batch.runtemplate.simin.MaxStep = max_events
            cur_batch.runtemplate.simin.SimTime_Max = 'inf'
            cur_batch.runtemplate.simin.WallTime_Max = 'inf'
            cur_batch.runtemplate.simin.restart = False
            
            cur_batch.runtemplate.simin.procstat = ['event', np.max( [max_events / n_samples, 1] ) ]
            cur_batch.runtemplate.simin.specnum = ['event', np.max( [max_events / n_samples, 1] ) ]
            cur_batch.runtemplate.simin.hist = ['event', np.max( [max_events * (n_samples-1) / n_samples, 1] )]       # only record the initial and final states

            SDF_vec = np.ones( cur_batch.runtemplate.mechin.get_num_rxns() )         # Initialize scaledown factors
        
        elif iteration > 1:             # Time sampling

            # Change sampling
            cur_batch.runtemplate.simin.MaxStep = 'inf'
            cur_batch.runtemplate.simin.WallTime_Max = 'inf'
            cur_batch.runtemplate.simin.restart = False
            cur_batch.runtemplate.simin.SimTime_Max = prev_batch.t_vec[-1] * scale_final_time
            cur_batch.runtemplate.simin.SimTime_Max = float('{0:.3E} \t'.format( cur_batch.runtemplate.simin.SimTime_Max ))     # round to 4 significant figures
            cur_batch.runtemplate.simin.procstat = ['time', cur_batch.runtemplate.simin.SimTime_Max / n_samples]
            cur_batch.runtemplate.simin.specnum = ['time', cur_batch.runtemplate.simin.SimTime_Max / n_samples]
            cur_batch.runtemplate.simin.hist = ['time', cur_batch.runtemplate.simin.SimTime_Max ]
            
            # Adjust pre-exponential factors based on the stiffness assessment of the previous iteration
            if include_stiff_reduc:
                cur_batch.runtemplate.AdjustPreExponentials(SDF_vec)
            
            # Use continuation
            initial_states = prev_batch.History_final_snaps
        
        # Run jobs and read output
        cur_batch.BuildJobFiles(init_states = initial_states)
        cur_batch.RunAllTrajectories_JobArray(server = 'Squidward', job_name = 'Iteration_' + str(iteration) )  
        cur_batch.ReadMultipleRuns()
        
        if iteration == 1:
            cum_batch = copy.deepcopy(cur_batch)
        else:
            cum_batch = append_replicates(prev_batch, cur_batch)         # combine with previous data
        
        # Test steady-state
        cum_batch.AverageRuns()
        acf_data = cum_batch.Compute_rate()
        
        sum_file.write( '\nIteration ' + str(iteration) )
        sum_file.write( 'Batches per trajectory: ' + str(cum_batch.Nbpt) )
        sum_file.write( 'Batch length (s): ' + str(cum_batch.batch_length) )
        sum_file.write( 'Rate: ' + str(cum_batch.rate) )
        sum_file.write( 'Rate confidence interval: ' + str(cum_batch.rate_CI) )

        
        # Test if enough product molecules have been produced
        enough_product = True
        
        # Test if rate is computed with sufficient accuracy
        if cum_batch.rate == 0:
            rate_accurate = False
        else:
            rate_accurate = (cum_batch.rate_CI / cum_batch.rate < rate_tol)
        
        sum_file.write( 'Decorrelated? ' + str(enough_product) )
        sum_file.write( 'Rate accurate? ' + str(rate_accurate) )
        sum_file.write( '\n' )
        
        is_steady_state = enough_product and rate_accurate
        # Terminate if the structure is inactive
        
        # Record information about the iteration
        cum_batch.runAvg.PlotGasSpecVsTime()
        cum_batch.runAvg.PlotSurfSpecVsTime()
        
        cur_batch.AverageRuns()
        cur_batch.runAvg.PlotElemStepFreqs()
        scaledown_data = ProcessStepFreqs(cur_batch.runAvg)         # compute change in scaledown factors based on simulation result
        delta_sdf = scaledown_data['delta_sdf']
        
        # Update scaledown factors
        for ind in range(len(SDF_vec)):
            SDF_vec[ind] = SDF_vec[ind] * delta_sdf[ind]
            
        scale_final_time = np.max( [1.0/np.min(delta_sdf), ss_inc] )
        
        prev_batch = copy.deepcopy(cum_batch)
        iteration += 1

    sum_file.close()
    return cum_batch
    

def ProcessStepFreqs(run, stiff_cut = 100.0, delta = 0.05, equilib_cut = 0.1):        # Change to allow for irreversible reactions
    
    '''
    Takes an average KMC trajectory and assesses the reaction frequencies to identify fast reactions
    Process KMC output and determine how to further scale down reactions
    Uses algorithm from A. Chatterjee, A.F. Voter, Accurate acceleration of kinetic Monte Carlo simulations through the modification of rate constants, J. Chem. Phys. 132 (2010) 194101.
    '''
    
    delta_sdf = np.ones( run.mechin.get_num_rxns() )    # initialize the marginal scaledown factors
    rxn_speeds = []
    
    # data analysis
    freqs = run.procstatout.events[-1,:]
    fwd_freqs = freqs[0::2]
    bwd_freqs = freqs[1::2]
    net_freqs = fwd_freqs - bwd_freqs
    tot_freqs = fwd_freqs + bwd_freqs
    
    fast_rxns = []
    slow_rxns = []        
    for i in range(len(tot_freqs)):
        if tot_freqs[i] == 0:
            slow_rxns.append(i)
            rxn_speeds.append('slow')
        else:
            PE = float(net_freqs[i]) / tot_freqs[i]
            if np.abs(PE) < equilib_cut:
                fast_rxns.append(i)
                rxn_speeds.append('fast')
            else:
                slow_rxns.append(i)
                rxn_speeds.append('slow')
    
    # Find slow scale rate
    slow_freqs = [1.0]      # put an extra 1 in case no slow reactions occur
    for i in slow_rxns:
        slow_freqs.append(tot_freqs[i])
    slow_scale = np.max(slow_freqs)
    
    # Adjust fast reactions closer to the slow scale
    for i in fast_rxns:
        N_f = tot_freqs[i] / float(slow_scale)              # number of fast events per rare event
        #alpha_UB = N_f * delta / np.log(1 / delta) + 1             # Chatterjee formula
        
        #delta_sdf[i] = np.min([1.0, np.max([stiff_cut / N_f, 1. / alpha_UB ]) ])
        delta_sdf[i] = np.min([1.0, stiff_cut / N_f ])
     
    return {'delta_sdf': delta_sdf, 'rxn_speeds': rxn_speeds, 'tot': tot_freqs, 'net': net_freqs}
    
    
def read_scaledown(RunPath):        # Need to add option for no scaledown
    
    '''
    Read KMC files from a scaledown that has already been run
    :returns: Cumulative replicates object with a group of trajectories
    '''

    n_folders = len(os.listdir(RunPath))            # Count the iterations

    cum_batch = None
    for ind in range(1,n_folders+1):
        
        x = zw.Replicates()
        x.ParentFolder = os.path.join(RunPath, 'Iteration_' + str(ind))
        x.ReadMultipleRuns()

        if ind == 1:
            cum_batch = x
        else:
            cum_batch = append_replicates(cum_batch, x)
            
    return cum_batch
    
    
def compute_site_rates(cat, kmc_reps, fldr, product = 'N2', gas_stoich = 1):
    
    '''
    Read structure occupancies and site propensities from a KMC folder
    :param cat: Catalyst structure
    :param kmc_reps: Replates object
    :param product: Gas phase product species
    :param gas_stoich: Stoichiometric coefficient of the product
    :returns: 
    '''
    
    kmc_reps.AverageRuns()
    avg_traj = kmc_reps.runAvg
    
    n_rxns = len( avg_traj.genout.RxnNameList )
    cat = NiPt_NH3()
    cat.load_defects(os.path.join(fldr_name,'structure.xsd'))
    site_props_ss = np.zeros( [cat.atoms_per_layer, n_rxns] )
    
    '''
    Identify stoichiometries
    '''
    
    # Find index of product molecule
    try:
        gas_prod_ind = len( avg_traj.simin.surf_spec ) + avg_traj.simin.gas_spec.index( gas_prod )           # Find the index of the product species and adjust index to account for surface species
    except:
        raise Exception('Gas species ' + gas_prod + ' not found.')

    # Find the stochiometry of the product molecule for each reaction
    nRxns = len(avg_traj.genout.RxnNameList)
    TOF_stoich = np.zeros(nRxns)
    for i, elem_stoich in enumerate(avg_traj.genout.Nu):
        TOF_stoich[i] = elem_stoich[gas_prod_ind]
    
    
    '''
    Fill in the rows of atoms that are present
    '''
    
    d_cut = 0.1     # Matches are not found if we lower this to 0.01
    
    mol_cart_coords = cat.atoms_template.get_positions()[:, 0:2:]
    
    for i in range(len(avg_traj.lat.site_type_inds)):
    
        lat_site_coords = avg_traj.lat.cart_coords[i, :]
        match_found = False
        
        
        defect_ind = -1
        while ( not match_found ) and ( defect_ind < len(cat.variable_atoms)-1 ):
            
            defect_ind += 1
            
            #if defect_ind >= len(cat.variable_atoms):          # Comment this out to only look for matches for top sites
            #    print lat_site_coords
            #    raise NameError('Molecular site not found for lattice site.')
        
            atom_ind = cat.variable_atoms[defect_ind]
            atom_pos = mol_cart_coords[atom_ind,:]
            
            match_found = ( np.linalg.norm( atom_pos - lat_site_coords ) < d_cut )
            
        if match_found:
            site_props_ss[defect_ind,:] = avg_traj.TS_site_props_ss[i,:]    
    
    site_rates = np.matmul( site_props_ss, TOF_stoich ) / gas_stoich
    np.save(os.path.join(fldr, 'site_rates.npy'), site_rates)
    
    return site_rates