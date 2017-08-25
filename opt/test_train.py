'''
Import data from an optimization and train a neural network to it
'''

import numpy as np
from plotpop import *

from NeuralNetwork import NeuralNetwork

X = np.load('X.npy')
Y = np.load('Y.npy')

surrogate = NeuralNetwork()
surrogate.refine( X, Y )
surrogate.plot_parity('parity.png', title = 'High fidelity data')
plot_pop_SO(Y, fname = 'High_fid_data.png', title = 'High fidelity data')