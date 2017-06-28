"""
Perform Neural Network fitting.
Note
- Make sure you have the latest scikit_learn. (For conda, simply update anaconda)
- This is not for a large model. Find some C++ package for large problem.
@author: Geun Ho Gu
"""
import os 
import numpy as np
import copy

from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error # don't judge me for being lazy


class NeuralNetwork():
    
    def __init__(self):
        
        self.X = None
        self.Y = None
        self.NNModel = None


    def predict(self, x):
        
        '''
        Compute value for the model as trained so far
        '''
        
        return self.NNModel.predict(x)
    
    
    def refine(self, X_plus, Y_plus):
        
        '''
        Add new data to the training set and retrain
        '''
        
        if self.X is None or self.Y is None:
            
            self.X = X_plus
            self.Y = Y_plus
            
        else:
            
            if len(self.X.shape) == 1:
                self.X = np.hstack( [ self.X , X_plus ] )
            else:
                self.X = np.vstack( [ self.X , X_plus ] )
                
            if len(self.Y.shape) == 1:
                self.Y = np.hstack( [ self.Y , Y_plus ] )
            else:
                self.Y = np.vstack( [ self.Y , Y_plus ] )
                
        self.train()


    def train(self):

        '''
        Train the neural network with the available data
        '''

        if self.X is None or self.Y is None:
            raise NameError('Data not defined.')

        '''
        User Input
        '''
        # data scaling: involves centering and standardizing data
        Data_scaling = True
        
        # Train:test split cross validation set up
        test_set_size = 0.2
        output_path = os.path.abspath('Neural_net_out.txt')
        
        # Neural network set up
        hidden_layer_sizes = (5,) # (#perceptons in layer 1, #perceptons in layer 2, #perceptons in layer 3, ...)
        regularization_parameters = np.logspace(-8, 3, num=100)

        '''
        Scale Data
        '''
        
        X = copy.deepcopy(self.X)
        Y = copy.deepcopy(self.Y)
        
        if Data_scaling:
            
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
            
        '''
        Perform regression
        '''
        
        ## Split Data
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_set_size, random_state=1)
        
        # output
        f = open(output_path,'w')
        for i in xrange(0,len(regularization_parameters)):
            
            ## Build Model
            NNModel = MLPRegressor(solver = 'lbfgs', alpha = regularization_parameters[i], hidden_layer_sizes = hidden_layer_sizes)
            
            ## Fit
            NNModel.fit(X_train, Y_train)
            
            ## Predict test set
            Y_test_predicted = NNModel.predict(X_test)
            
            ## Compute mean square error
            f.write( '%.5e\t%.5e\n'%(regularization_parameters[i], mean_squared_error(Y_scaler.inverse_transform(Y_test), 
                                     Y_scaler.inverse_transform(Y_test_predicted))))
            
        f.close()