'''
Regress a neural network to active site local environment rather than structures
'''

import numpy as np

from cat_optimization.neural_net import NeuralNetwork
from ORR import orr_cat
from NH3.NiPt_NH3 import NiPt_NH3


eval_obj = NiPt_NH3()
#eval_obj = orr_cat()
eval_obj.randomize(coverage = 0.7, build_structure = True)
#eval_obj.load_from_file('SA_opt.xsd')
eval_obj.show(fname = 'random_structure', fmat = 'png')
curr_list = eval_obj.get_site_data()
trans_mat = eval_obj.generate_all_translations()

act_norm = np.max(curr_list)

neural_net = NeuralNetwork(verbose=True, learning_rate_init=0.0001,
                alpha = 1.0, hidden_layer_sizes = (81,), max_iter=10000, tol=0.00001)
neural_net.refine( trans_mat, curr_list / act_norm )
neural_net.plot_parity()

'''
Use that data to evaluate a different structure
'''

#test_cat = cat_optimize()
#test_cat.randomize(coverage = 0.7, build_graph = True)
#
#high_fid = eval_obj.eval_current_density()
#high_fid2 = test_cat.eval_current_density()
#
## Predict site-wise activities with neural network
#site_currents = act_norm * neural_net.predict(trans_mat).flatten()
#low_fid = sum(site_currents) / ( eval_obj.surface_area * 1.0e-16)
#
#trans_mat2 = test_cat.generate_all_translations()
#site_currents2 = act_norm * neural_net.predict(trans_mat2).flatten()
#low_fid2 = sum(site_currents2) / ( test_cat.surface_area * 1.0e-16)
#
#print high_fid
#print low_fid
#
#print high_fid2
#print low_fid2