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
from sklearn.metrics import mean_squared_error


class NeuralNetwork():
    
    def __init__(self):
        
        self.X = None           # numpy array of x values
        self.Y = None           # numpy array of y values
        self.Y_nn = None        # y values predicted by the neural network
        self.NNModel = None


    def predict(self, x):
        
        '''
        Compute value for the model as trained so far
        '''

        return self.NNModel.predict( x )
    
    
    def refine(self, X_plus, Y_plus):
        
        '''
        Add new data to the training set and retrain
        X_plus and Y_plus need to be 2-D matrices of data
        '''
        
        # Append to existing data sets
        if self.X is None or self.Y is None:
            self.X = X_plus
            self.Y = Y_plus
        else:
            self.X = np.vstack( [ self.X , X_plus ] )
            self.Y = np.vstack( [ self.Y , Y_plus ] )
        
        # Regress the neural network
        self.train()
        self.Y_nn = self.NNModel.predict( self.X )      # store predicted values


    def plot_parity(self, fname = 'parity.png', logscale = 'False'):
    
        '''
        Plot a parity plot for the data trained so far
        '''
    
        mat.rcParams['mathtext.default'] = 'regular'
        mat.rcParams['text.latex.unicode'] = 'False'
        mat.rcParams['legend.numpoints'] = 1
        mat.rcParams['lines.linewidth'] = 2
        mat.rcParams['lines.markersize'] = 12
        
        plt.figure()
        plt.plot(self.Y[:,1], self.Y_nn[:,1], 'o')
        
        #plt.xticks(size=20)
        #plt.yticks(size=20)
        plt.xlabel('High fidelity', size=24)
        plt.ylabel('Neural network', size=24)
        #plt.legend(series_labels, loc=4, prop={'size':20}, frameon=False)
        plt.tight_layout()
        
        if logscale:
            plt.yscale('log')
        
        plt.savefig(fname)
        plt.close()
        
        
    def train(self, regularization_parameter = 0.1):

        '''
        Train the neural network with the available data
        '''

        if self.X is None or self.Y is None:
            raise NameError('Data not defined.')

        '''
        User Input
        '''
        # data scaling: involves centering and standardizing data
#        Data_scaling = True
        Data_scaling = False
        
        # Train:test split cross validation set up
        test_set_size = 0.2
        
        # Neural network set up
        hidden_layer_sizes = (20,) # (#perceptrons in layer 1, #perceptrons in layer 2, #perceptrons in layer 3, ...)

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
            
        ## Build Model
        self.NNModel = MLPRegressor(activation = 'logistic', solver = 'lbfgs', alpha = regularization_parameter, hidden_layer_sizes = hidden_layer_sizes)
        
        ## Fit
        self.NNModel.fit(X_train, Y_train)