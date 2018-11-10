#!/opt/shared/python/2.7.8-1/bin/python

'''
Read database and try to regress a surrogate model
'''

import os
import sys
sys.path.append('/home/vlachos/mpnunez/Github/Zacros-Wrapper')
sys.path.append('/home/vlachos/mpnunez/Github/Structure-Optimization')
sys.path.append('/home/vlachos/mpnunez/python_packages/ase')
sys.path.append('/home/vlachos/mpnunez/python_packages/sklearn/lib/python2.7/site-packages')

import numpy as np
import matplotlib as mat
import matplotlib.pyplot as plt

# Import sklearn classes
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import OML

# Get command line options
import argparse
parser = argparse.ArgumentParser(description='Analyze a database and do a machine learning fit')
parser.add_argument('-l', action='store_true',help='Use local occupancies only')
parser.add_argument('-w', action='store_true',help='Weight the data points')
parser.add_argument('-z', action='store_true',help='Include sites with zero rate')
args = parser.parse_args()
raise NameError('stop')
# Load database files - Needs to read it in a loop
occ_DB_list = []
KMC_site_type_DB_list = []
site_rates_DB_list = []
for i in range(96):
#for i in range(10):
    #print 'Reading database ' + str(i)
    occ_DB_list.append( np.load( os.path.join('data','occ_DB_' + str(i+1) + '.npy' )) )    
    KMC_site_type_DB_list.append( np.load(os.path.join('data','KMC_site_type_DB_' + str(i+1) + '.npy') ))
    site_rates_DB_list.append( np.load(os.path.join('data','site_rates_DB_' + str(i+1) + '.npy') ))
    
    #print occ_DB_list[0].shape
    #print KMC_site_type_DB_list[0].shape
    #print site_rates_DB_list[0].shape
    #raise NameError('stop')
    
occ_DB = np.vstack(occ_DB_list)
KMC_site_type_DB = np.vstack(KMC_site_type_DB_list)
site_rates_DB = np.hstack(site_rates_DB_list)
    
# Use only local occupancies
if args.l:
    cat = OML.LSC_cat()
    occ_DB = occ_DB[:,cat.get_local_inds()]

'''
Shape the data for regression
'''

y_max = np.max(site_rates_DB)
print y_max
if y_max == 0.:
    raise NameError('No active sites in database.')

X_reg = []
Y_reg = []
ndp = len(site_rates_DB)
site_is_active = np.zeros_like(site_rates_DB)

n_active = 0
for index in range(len(site_rates_DB)):
    
    if site_rates_DB[index] > 0.0:
        X_reg.append( occ_DB[index,:] )
        #X_reg.append( KMC_site_type_DB[index,:] )
        Y_reg.append( site_rates_DB[index] )
        site_is_active[index] = 1     # Classify rates as zero or nonzero
        n_active += 1
    else:
        if args.l:
            X_reg.append( occ_DB[index,:] )
            Y_reg.append( site_rates_DB[index] )
        


print str(ndp) + ' total sites'
print str(n_active) + ' active sites'
print str(ndp - n_active) + ' inactive sites'

X_reg = np.array(X_reg)
Y_reg = np.array(Y_reg)

# Assign weights to data points
weights = np.ones_like(Y_reg)
if args.w:
    for index in range(len(Y_reg)):
        if site_is_active[index] == 1:
            weights[index] = float(ndp - n_active) / ndp
        else:
            weights[index] = float(n_active) / ndp

'''
Regress nonzero site rates
'''

plt.hist(Y_reg, weights=weights)
plt.xlabel('Rate (s^-1)', size=24)
plt.ylabel('Frequency', size=24)
plt.xticks(size=20)
plt.yticks(size=20)
plt.tight_layout()
plt.savefig('site_rate_hist')
plt.close()
#raise NameError('stop')

# Scale Y
Y_scaler = StandardScaler(with_mean = True, with_std = True)
Y_scaler.fit(Y_reg.reshape(-1,1))
Y = Y_scaler.transform(Y_reg.reshape(-1,1))
Y = Y.flatten()

'''
Train neural network - Regression
'''

# Split the training and test sets
X_train, X_test, y_train, y_test, weights_train, weights_test = train_test_split(X_reg, Y, weights, random_state=0)
y_train_raw = Y_scaler.inverse_transform(y_train)
y_test_raw = Y_scaler.inverse_transform(y_test)

# Neural network
#predictor = MLPRegressor(activation = 'relu', verbose=True, learning_rate_init=0.0001,
#                alpha = 0.1, hidden_layer_sizes = (20,), max_iter=10000, tol=0.00001) 

predictor = tree.DecisionTreeRegressor(max_depth=10)
predictor.fit(X_train, y_train, sample_weight=weights_train)

# Analyze performance on training set
print 'Mean y: ' + str(np.mean(Y_reg))
predictions_train = Y_scaler.inverse_transform( predictor.predict(X_train) )
predictions_train = predictions_train.reshape(-1)
y_train_raw = y_train_raw.reshape(-1)
mae = np.mean( np.abs( predictions_train - y_train_raw ) )
print 'Mean absolute error (train): ' + str(mae)

# Analyze performance on test set
predictions_test = Y_scaler.inverse_transform( predictor.predict(X_test) )
predictions_test = predictions_test.reshape(-1)
y_test_raw = y_test_raw.reshape(-1)
mae = np.mean( np.abs( predictions_test - y_test_raw ) )
print 'Mean absolute error (test): ' + str(mae)

print str(len(y_train_raw)) + ' data points in training set'
print str(len(y_test_raw)) + ' data points in test set'

'''
Analyze decision tree structure
'''

# Using those arrays, we can parse the tree structure:

estimator = predictor   # Change the name because we are copy/pasting code

n_nodes = estimator.tree_.node_count
children_left = estimator.tree_.children_left
children_right = estimator.tree_.children_right
feature = estimator.tree_.feature
threshold = estimator.tree_.threshold


# The tree structure can be traversed to compute various properties such
# as the depth of each node and whether or not it is a leaf.
node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
is_leaves = np.zeros(shape=n_nodes, dtype=bool)
stack = [(0, -1)]  # seed is the root node id and its parent depth
while len(stack) > 0:
    node_id, parent_depth = stack.pop()
    node_depth[node_id] = parent_depth + 1

    # If we have a test node
    if (children_left[node_id] != children_right[node_id]):
        stack.append((children_left[node_id], parent_depth + 1))
        stack.append((children_right[node_id], parent_depth + 1))
    else:
        is_leaves[node_id] = True

print("The binary tree structure has %s nodes and has "
      "the following tree structure:"
      % n_nodes)
for i in range(n_nodes):
    if is_leaves[i]:
        print("%snode=%s leaf node." % (node_depth[i] * "\t", i))
    else:
        print("%snode=%s test node: go to node %s if X[:, %s] <= %s else to "
              "node %s."
              % (node_depth[i] * "\t",
                 i,
                 children_left[i],
                 feature[i],
                 threshold[i],
                 children_right[i],
                 ))
print()

'''
Plot results
'''

mat.rcParams['mathtext.default'] = 'regular'
mat.rcParams['text.latex.unicode'] = 'False'
mat.rcParams['legend.numpoints'] = 1
mat.rcParams['lines.linewidth'] = 2
mat.rcParams['lines.markersize'] = 12

plt.figure()
all_point_values = np.hstack([y_train_raw, y_test_raw, predictions_train, predictions_test])
par_min = min( all_point_values )
par_max = max( all_point_values )
plt.plot( [par_min, par_max], [par_min, par_max], '--', color = 'k')  # Can do this for all outputs
plt.plot(y_train_raw, predictions_train, 'o', color = 'b', label = 'train')
plt.plot(y_test_raw, predictions_test, 'x', color = 'r', label = 'test')

plt.xticks(size=18)
plt.yticks(size=18)
plt.xlabel('Kinetic Monte Carlo', size=24)
plt.ylabel('Surrogate model', size=24)
plt.legend(loc=4, prop={'size':20}, frameon=False)
plt.tight_layout()
plt.savefig('hyper_fit', dpi = 600)
plt.close()