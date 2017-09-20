'''
Import data from an optimization and train a neural network to it
'''

import sys
sys.path.append('/home/vlachos/mpnunez/Github/Structure-Optimization/ORR/')
sys.path.append('/home/vlachos/mpnunez/Github/Zacros-Wrapper')
sys.path.append('/home/vlachos/mpnunez/python_packages/ase')
sys.path.append('/home/vlachos/mpnunez/python_packages/sklearn/lib/python2.7/site-packages')
sys.path.append('/home/vlachos/mpnunez/python_packages/DEAP/lib/python2.7/site-packages')

import numpy as np
from plotpop import *

from NeuralNetwork import NeuralNetwork

if __name__ == '__main__':

    X = np.load('X.npy')
    Y = np.load('Y.npy')
    
    print X.shape
    print Y.shape
    #raise NameError('stop')
    surrogate = NeuralNetwork()
    surrogate.X = X
    surrogate.Y = Y.reshape(-1)
    
    reg_param_vec = np.logspace(-8, 3, num=20)
    CV_score_vec = []
    
    for lambda_ in reg_param_vec:
        CV_score_vec.append( surrogate.k_fold_CV(reg_param = lambda_) )
        
    print reg_param_vec
    print CV_score_vec