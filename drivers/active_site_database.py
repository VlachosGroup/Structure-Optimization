'''
Regress a neural network to active site local environment rather than structures
'''

import numpy as np

from cat_optimize import cat_optimize
from NeuralNetwork import NeuralNetwork

eval_obj = cat_optimize()
eval_obj.load_from_file('SA_opt.xsd')

curr_list = eval_obj.get_site_currents()
curr_list = np.transpose( np.array(curr_list).reshape([2,144]) )

trans_mat = eval_obj.generate_all_translations()

#print trans_mat                 # X for the neural network
#print curr_list                 # Y for the neural network

act_norm = np.max(curr_list)

#print curr_list / act_norm

#raise NameError('stop')

neural_net = NeuralNetwork()
neural_net.refine( trans_mat, curr_list / act_norm )
neural_net.plot_parity()