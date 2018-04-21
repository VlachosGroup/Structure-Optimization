# Read the output for a Zacros trajectory and plot data

import zacros_wrapper as zw
import os


db_fldr = '/home/vlachos/mpnunez/OML_data/NH3_data_1/KMC_DB'
fldr_list = [os.path.join(db_fldr, o) for o in os.listdir(db_fldr) if os.path.isdir(os.path.join(db_fldr,o))]

for RunPath in fldr_list:

    if not os.path.isfile(os.path.join(RunPath, 'general_output.txt')):
        continue

    print RunPath

    ''' Read simulation results '''
    my_trajectory = zw.kmc_traj()                           # Create single trajectory object
    my_trajectory.Path = RunPath                            # Set directory for files
    my_trajectory.ReadAllOutput(build_lattice=True)         # Read input and output files
    
    ''' Plot data '''
    my_trajectory.PlotSurfSpecVsTime()                      # Plot surface species populations versus time
    my_trajectory.PlotGasSpecVsTime()                       # Plot gas species populations versus time
    try:
        my_trajectory.PlotElemStepFreqs(time_norm = False)       # Plot elementary step frequencies
    except:
        continue