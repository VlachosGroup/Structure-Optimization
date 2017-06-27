"""
Perform Neural Network fitting.
Note
- Make sure you have the latest scikit_learn. (For conda, simply update anaconda)
- This is not for a large model. Find some C++ package for large problem.
@author: Geun Ho Gu
"""
import os 
import numpy as np
from sklearn.neural_network import MLPRegressor
################################# User Input ##################################
# file path
Xpath = os.path.abspath(r'.\finger_list.pickle')
Ypath = os.path.abspath(r'.\rate.pickle')
#Xpath = os.path.abspath(r'.\Descriptor.yaml')
#Ypath = os.path.abspath(r'.\Response.yaml')
# data scaling: involves centering and standardizing data
Data_scaling = True
# Train:test split cross validation set up
test_set_size = 0.2
output_path = os.path.abspath(r'.\output.yaml')
# Neural network set up
hidden_layer_sizes = (5,) # (#perceptons in layer 1, #perceptons in layer 2, #perceptons in layer 3, ...)
regularization_parameters = np.logspace(-8, 3, num=100)
################################# User Input ##################################
# Load up data
if Xpath[-6:] == 'pickle':
    import pickle
    X = pickle.load(file(Xpath,'r'))
    Y = pickle.load(file(Ypath,'r'))
elif Xpath[-4:] == 'yaml':
    import yaml
    X = yaml.load(file(Xpath,'r'))
    Y = yaml.load(file(Ypath,'r'))
X = np.array(X,dtype=float)
Y = np.array(Y,dtype=float)
Y = Y.reshape(-1,1)
# Scale Data
if Data_scaling:
    from sklearn.preprocessing import StandardScaler
    # Scaling options
    centering = True
    standardization = True
    # scaling X
    X_scaler = StandardScaler(with_mean = centering, with_std = standardization)
    X_scaler.fit(X)
    X = X_scaler.transform(X)
    # scaling Y
    Y_scaler = StandardScaler(with_mean = centering, with_std = standardization)
    Y_scaler.fit(Y)
    Y = Y_scaler.transform(Y)
    Y = Y.reshape(-1)
# Perform regression 
## Split Data
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(\
    X, Y, test_size=test_set_size, random_state=1)
# output
f = open(output_path,'w')
for i in xrange(0,len(regularization_parameters)):
    ## Build Model
    NNModel = MLPRegressor(solver='lbfgs',alpha=regularization_parameters[i], 
                       hidden_layer_sizes=hidden_layer_sizes)
    ## Fit
    NNModel.fit(X_train,Y_train)
    ## Predict test set
    Y_test_predicted = NNModel.predict(X_test)
    ## Compute mean square error
    from sklearn.metrics import mean_squared_error # don't judge me for being lazy
    f.write('%.5e\t%.5e\n'%(regularization_parameters[i],mean_squared_error(Y_scaler.inverse_transform(Y_test),Y_scaler.inverse_transform(Y_test_predicted))))
f.close()