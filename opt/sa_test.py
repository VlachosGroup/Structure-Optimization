'''
Build a catalyst surface and optimize it using simulated annealing
'''

import sys
sys.path.append('/home/vlachos/mpnunez/Github/Structure-Optimization/ORR/')
from Cat_structure import cat_structure
from cat_optimize import cat_optimize
from sim_anneal import *
from metal import metal
from ORR import ORR_rate

# Initialize catalyst variable
x = cat_optimize()
x.randomize()

# Find activity normalization
Pt_met = metal('Pt')
active_norm = ORR_rate(Pt_met.get_OH_BE(8.27), Pt_met.get_OOH_BE(8.27)) / ( x.surface_area * 1.0e-16)
x.weights = [0, -1.]

# Optimize
optimize(x, total_steps = 500 * 144, initial_T = 0.7 * active_norm, n_record = 100)
print str(x.eval_current_density()) + ' mA/cm^2'

x.show(fmat = 'picture')
x.show(fmat = '.xsd')