'''
Read database and try to regress a surrogate model
'''

import sys
sys.path.append('/home/vlachos/mpnunez/Github/Zacros-Wrapper')
sys.path.append('/home/vlachos/mpnunez/Github/Structure-Optimization')
sys.path.append('/home/vlachos/mpnunez/python_packages/ase')
sys.path.append('/home/vlachos/mpnunez/python_packages/sklearn/lib/python2.7/site-packages')

import numpy as np
import matplotlib as mat
import matplotlib.pyplot as plt
mat.rcParams['mathtext.default'] = 'regular'
mat.rcParams['text.latex.unicode'] = 'False'
mat.rcParams['legend.numpoints'] = 1
mat.rcParams['lines.linewidth'] = 2
mat.rcParams['lines.markersize'] = 12

from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import math


def f(x,noise = 0.5):
    #return np.sin(2 * math.pi * x)
    x_sqr = x ** 2
    #return x_sqr + noise * np.random.normal()
    if noise == 0:
        return x_sqr
    else:
        return np.random.poisson(lam=x_sqr)
    
'''
Generate data
'''

n_data = 1000
X = np.linspace(0,3,n_data)
Y = np.zeros_like(X)
Y_nonoise = np.zeros_like(X)
for i in xrange(n_data):
    Y[i] = f(X[i],noise = 0)        # let's just do a regular function for now
    Y_nonoise[i] = f(X[i],noise=0)

'''
Regress
'''

# Split data into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=0)

# Scale Y training data
Y_scaler = StandardScaler(with_mean = True, with_std = True)
Y_scaler.fit(y_train.reshape(-1,1))
y_train_s = Y_scaler.transform(y_train.reshape(-1,1))
y_test_s = Y_scaler.transform(y_test.reshape(-1,1))

X_scaler = StandardScaler(with_mean = True, with_std = True)
X_scaler.fit(X_train.reshape(-1,1))
X_train_s = X_scaler.transform(X_train.reshape(-1,1))
X_test_s = X_scaler.transform(X_test.reshape(-1,1))

# Create the model and fit
predictor = tree.DecisionTreeRegressor(max_depth=4)
predictor.fit(X_train_s, y_train_s)

# Evaluate using the model
y_train_model = Y_scaler.inverse_transform( predictor.predict(X_train_s) )
y_test_model = Y_scaler.inverse_transform( predictor.predict(X_test_s) )


'''
Show the fit
'''

plt.figure()
plt.plot(X, Y_nonoise, '-', color = 'k', label = 'data')
#plt.plot(X_train, y_train_model, 'o', color = 'b', label = 'train')
#plt.plot(X_test, y_test_model, 'o', color = 'r', label = 'test')
plt.plot(X_train, y_train, 'o', color = 'b', label = 'train')
plt.plot(X_test, y_test, 'o', color = 'r', label = 'test')
plt.xticks(size=18)
plt.yticks(size=18)
plt.xlabel('x', size=24)
plt.ylabel('y', size=24)
plt.legend(loc=2, prop={'size':20}, frameon=False)
plt.tight_layout()
plt.savefig('decision_tree', dpi = 600)
plt.close()

all_data = np.hstack([y_train, y_train_model,y_test, y_test_model])
y_min = np.min(all_data)
y_max = np.max(all_data)

plt.figure()
plt.plot(np.array([y_min,y_max]), np.array([y_min,y_max]), '--', color = 'k')
plt.plot(y_train, y_train_model, 'o', color = 'b', label = 'train')
plt.plot(y_test, y_test_model, 'o', color = 'r', label = 'test')
plt.xticks(size=18)
plt.yticks(size=18)
plt.xlabel('data', size=24)
plt.ylabel('Decision tree', size=24)
plt.legend(loc=4, prop={'size':20}, frameon=False)
plt.tight_layout()
plt.savefig('parity', dpi = 600)
plt.close()

'''
Analyze decision tree structure
'''