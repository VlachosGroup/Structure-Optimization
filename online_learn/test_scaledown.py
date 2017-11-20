'''
Test evaluating a structure with scaledown
'''

import os

from NH3.NiPt_NH3 import NiPt_NH3
from RateRescaling import *
import zacros_wrapper as zw
from zacros_wrapper.utils import *


'''
Make this a function with some input variables
:returns: A replicates object for the cumulative trajectories
'''

# Directories
structure_folder = '/home/vlachos/mpnunez/OML_data/NH3_data_1/KMC_DB/structure_210'
run_folder = '/home/vlachos/mpnunez/OML_data/test_scaledown'
kmc_input_folder = '/home/vlachos/mpnunez/OML_data/NH3_data_1/KMC_input'

# Clear folder
if not os.path.exists(run_folder):
    os.makedirs(run_folder)
ClearFolderContents(run_folder)

# Create KMC trajectory object
kmc_traj = zw.kmc_traj()
kmc_traj.simin.ReadIn(kmc_input_folder)
kmc_traj.mechin.ReadIn(kmc_input_folder)
kmc_traj.clusterin.ReadIn(kmc_input_folder)
kmc_traj.Path = run_folder
kmc_traj.gas_prod = 'N2'
kmc_traj.exe_file = '/home/vlachos/mpnunez/bin/zacros_ML.x'

# Create catalyst structure and lattice
cat = NiPt_NH3()
cat.load_defects(os.path.join(structure_folder, 'structure.xsd'))
cat.graph_to_KMClattice()
kmc_traj.lat = cat.KMC_lat          # assign structure lattice to kmc template

# Write out files for catalyst lattice
cat.show(fname = os.path.join(run_folder,'structure'), fmat = 'png')
cat.show(fname = os.path.join(run_folder,'structure'), fmat = 'xsd')
kmc_lat = cat.KMC_lat.PlotLattice()
kmc_lat.savefig(os.path.join(run_folder,'lattice.png'),format='png', dpi=600)
kmc_lat.close()

# Run the rate constant rescaling
x = ReachSteadyStateAndRescale(kmc_traj, run_folder, n_runs = 16, n_batches = 1000, 
    prod_cut = 1000, include_stiff_reduc = True, max_events = int(1e2), 
    max_iterations = 2, ss_inc = 1.0, n_samples = 100, parallel_mode = 'Squidward',
    rate_tol = 0.05)
    
print x.runAvg.TS_site_props_ss