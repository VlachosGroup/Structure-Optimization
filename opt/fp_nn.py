'''
Testing out a "first principles" neural network where we hand-specify the weights and intercepts
'''

import random
import numpy as np

import sys
sys.path.append('/home/vlachos/mpnunez/Github/Structure-Optimization/ORR/')
from Cat_structure import cat_structure

from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt
import matplotlib as mat
import matplotlib.ticker as mtick

'''
Define objective function
'''

eval_obj = cat_structure(met_name = 'Pt', facet = '111', dim1 = 4, dim2 = 4)
def evalFitness_HF(individual):
    return eval_obj.eval_x( individual )[0]

# Initialize population and evaluate
population = [ [0 for i in xrange(16)] for j in xrange(100) ]

# Randomize according to coverages
for i in xrange(len(population)):
    coverage = float(i) / len(population[i])
    for j in range(len(population[i])):
        if random.random() < coverage:
            population[i][j] = 1
        else:
            population[i][j] = 0

X = np.array(population)
Y = [ evalFitness_HF( population[i] ) for i in xrange(len(population)) ]
Y = np.array(Y)
     
#print eval_obj.template_graph.vertices()
#print eval_obj.template_graph.edges()
print eval_obj.variable_atoms



'''
Define neural network
'''

neural_net = MLPRegressor(activation = 'relu', verbose=True, learning_rate_init=0.01,
                alpha = 0.01, hidden_layer_sizes = (32,))

neural_net.fit(X,Y)
                

    
# Hidden layer intercepts
for i in range(16):
    neural_net.intercepts_[0][i] = 3
    
for i in range(16,32):
    neural_net.intercepts_[0][i] = -91.

# Output layer intercepts    
neural_net.intercepts_[1][0] = 0

# Hidden layer weights
for i in range(16):
    for var_atom in eval_obj.variable_atoms:
        if var_atom in eval_obj.template_graph.get_neighbors(i+32):
            neural_net.coefs_[0][var_atom-48,i] = -1
        else:
            neural_net.coefs_[0][var_atom-48,i] = 0

for i in range(16,32):
    
    for var_atom in eval_obj.variable_atoms:
        if var_atom in eval_obj.template_graph.get_neighbors(i+32):
            neural_net.coefs_[0][var_atom-48,i] = -1
        else:
            neural_net.coefs_[0][var_atom-48,i] = 0
            
    neural_net.coefs_[0][i-16,i] = 100
    
# Output layer weights
for i in xrange(32):
    neural_net.coefs_[1][i] = eval_obj.metal.E_coh / 12.

for weight_mat in neural_net.coefs_:
    print weight_mat
    
for inter in neural_net.intercepts_:
    print inter
    
    
Y_pred = neural_net.predict(X)

mat.rcParams['mathtext.default'] = 'regular'
mat.rcParams['text.latex.unicode'] = 'False'
mat.rcParams['legend.numpoints'] = 1
mat.rcParams['lines.linewidth'] = 2
mat.rcParams['lines.markersize'] = 12

plt.figure()
plt.plot(Y, Y_pred, 'o')  # Can do this for all outputs
par_min = min( [ np.min(Y), np.min(Y_pred)] )
par_max = max( [ np.max(Y), np.max(Y_pred)] )
plt.plot( [par_min, par_max], [par_min, par_max], '-', color = 'k')  # Can do this for all outputs

plt.xticks(size=18)
plt.yticks(size=18)
plt.xlabel('High fidelity', size=24)
plt.ylabel('Neural network', size=24)
#plt.xlim([1.4,2.6])
#plt.ylim([1.4,2.6])
#plt.legend(series_labels, loc=4, prop={'size':20}, frameon=False)
plt.tight_layout()
plt.savefig('first_principles.png')
plt.close()