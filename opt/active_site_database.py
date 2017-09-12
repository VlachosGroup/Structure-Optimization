'''
Regress a neural network to active site local environment rather than structures
'''

from cat_optimize import cat_optimize
from NeuralNetwork import NeuralNetwork

eval_obj = cat_optimize()
x = eval_obj.randomize()
eval_obj.eval_x(x)
eval_obj.show(fmat = 'povray')