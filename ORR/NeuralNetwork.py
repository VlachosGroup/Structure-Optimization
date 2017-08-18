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

import matplotlib.pyplot as plt
import matplotlib as mat
import matplotlib.ticker as mtick

class NeuralNetwork():
    
    def __init__(self):
        
        self.X = None           # numpy array of x values
        self.Y = None           # numpy array of y values
        self.Ymeans = None
        self.Ystds = None
        
        self.Y_nn = None        # y values predicted by the neural network
        self.NNModel = None


    def predict(self, x):
        
        '''
        Compute value for the model as trained so far
        '''

        Ypred = self.NNModel.predict( x )
    
        for i in range(len(self.Ymeans)):
            Ypred[:,i] = self.Ystds[i] * Ypred[:,i] + self.Ymeans[i] * np.ones(Ypred[:,i].shape)
            
        return Ypred
    
    
    def refine(self, X_plus, Y_plus):
        
        '''
        Add new data to the training set and retrain
        X_plus and Y_plus need to be 2-D matrices of data
        '''
        
        self.NNModel = None
        
        # Append to existing data sets
        if self.X is None or self.Y is None:
            self.X = X_plus
            self.Y = Y_plus
        else:
            self.X = np.vstack( [ self.X , X_plus ] )
            self.Y = np.vstack( [ self.Y , Y_plus ] )
        
        # Regress the neural network
        self.train()
        self.Y_nn = self.predict( self.X )      # store predicted values


    def plot_parity(self, fname = 'parity.png', title = None, logscale = False):
    
        '''
        Plot a parity plot for the data trained so far
        '''
    
        mat.rcParams['mathtext.default'] = 'regular'
        mat.rcParams['text.latex.unicode'] = 'False'
        mat.rcParams['legend.numpoints'] = 1
        mat.rcParams['lines.linewidth'] = 2
        mat.rcParams['lines.markersize'] = 12
        
        plt.figure()
        plt.plot(self.Y[:,0], self.Y_nn[:,0], 'o')  # Can do this for all outputs
        par_min = np.min( np.vstack([self.Y[:,0], self.Y_nn[:,0]]) )
        par_max = np.max( np.vstack([self.Y[:,0], self.Y_nn[:,0]]) )
        plt.plot( [par_min, par_max], [par_min, par_max], '-', color = 'k')  # Can do this for all outputs
        
        plt.xticks(size=18)
        plt.yticks(size=18)
        plt.xlabel('High fidelity', size=24)
        plt.ylabel('Neural network', size=24)
#        plt.xlim([1.4,2.6])
#        plt.ylim([1.4,2.6])
        if not title is None:
            plt.title(title, size = 24)
        #plt.legend(series_labels, loc=4, prop={'size':20}, frameon=False)
        plt.tight_layout()
        
        if logscale:
            plt.yscale('log')
        
        plt.savefig('Y1_' + fname)
        plt.close()
        
        plt.figure()
        plt.plot(self.Y[:,1], self.Y_nn[:,1], 'o')  # Can do this for all outputs
        
        par_min = np.min( np.vstack([self.Y[:,1], self.Y_nn[:,1]]) )
        par_max = np.max( np.vstack([self.Y[:,1], self.Y_nn[:,1]]) )
        plt.plot([par_min, par_max], [par_min, par_max], '-', color = 'k')
        
        plt.xticks(size=18)
        plt.yticks(size=18)
        plt.xlabel('High fidelity', size=24)
        plt.ylabel('Neural network', size=24)
#        plt.xlim([-70,0])
#        plt.ylim([-70,0])
        if not title is None:
            plt.title(title, size = 24)
        #plt.legend(series_labels, loc=4, prop={'size':20}, frameon=False)
        plt.tight_layout()
        
        if logscale:
            plt.yscale('log')
        
        plt.savefig('Y2_' + fname)
        plt.close()
        
        
    def train(self, regularization_parameter = 10.0):

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
        
        # Train:test split cross validation set up
        test_set_size = 0.2
        
        # Neural network set up
        #N_nodes = self.X.shape[1]
        hidden_layer_sizes = (144,) # (#perceptrons in layer 1, #perceptrons in layer 2, #perceptrons in layer 3, ...)

        '''
        Scale Data
        '''
        
        X = copy.deepcopy(self.X)
        Y = copy.deepcopy(self.Y)
        
        # Normalize the y values
        self.Ymeans = []
        self.Ystds = []
        for i in range(Y.shape[1]):
            self.Ymeans.append( np.mean( Y[:,i] ) )
            self.Ystds.append( np.std( Y[:,i] ) )
            
            Y[:,i] = Y[:,i] - self.Ymeans[-1] * np.ones(Y[:,i].shape)
            
            # Divide by standard deviation if it is greater than zero
            if self.Ystds[-1] > 0:
                Y[:,i] = Y[:,i] / self.Ystds[-1]
        
        
        '''
        Perform regression
        '''
        
        ## Split Data
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_set_size, random_state=1)
            
        ## Build Model
        self.NNModel = MLPRegressor(activation = 'relu', solver = 'lbfgs', alpha = regularization_parameter, hidden_layer_sizes = hidden_layer_sizes)
        
        ## Fit
#        self.NNModel.fit(X_train, Y_train)
        self.NNModel.fit(X_train, Y_train)
        
        # Analyze neural network fit
        for weight_mat in self.NNModel.coefs_:
            print weight_mat
        
        print 'Trained neural network with ' + str(X.shape[0]) + ' data points.'
        self.Y_nn = self.predict( self.X )
        Y_train_nn = self.predict( X_train )      # store predicted values
        Y_test_nn = self.predict( X_test )      # store predicted values
        
        # Transform back to original variables
        Y_train[:,0] = self.Ystds[0] * Y_train[:,0] + self.Ymeans[0] * np.ones(Y_train[:,0].shape)
        Y_train[:,1] = self.Ystds[1] * Y_train[:,1] + self.Ymeans[1] * np.ones(Y_train[:,1].shape)
        
        Y_test[:,0] = self.Ystds[0] * Y_test[:,0] + self.Ymeans[0] * np.ones(Y_test[:,0].shape)
        Y_test[:,1] = self.Ystds[1] * Y_test[:,1] + self.Ymeans[1] * np.ones(Y_test[:,1].shape)
        
        '''
        Plot parity plots for analysis
        '''
        
        mat.rcParams['mathtext.default'] = 'regular'
        mat.rcParams['text.latex.unicode'] = 'False'
        mat.rcParams['legend.numpoints'] = 1
        mat.rcParams['lines.linewidth'] = 2
        mat.rcParams['lines.markersize'] = 12
        
        
        
        plt.figure()
        plt.plot(Y_train[:,1] , Y_train_nn[:,1], 'o', color = 'b')  # Can do this for all outputs
        plt.plot(Y_test[:,1], Y_test_nn[:,1], 'o', color = 'r')
        
        par_min = np.min( np.vstack([self.Y[:,1], self.Y_nn[:,1]]) )
        par_max = np.max( np.vstack([self.Y[:,1], self.Y_nn[:,1]]) )
        plt.plot([par_min, par_max], [par_min, par_max], '-', color = 'k')
        
        plt.xticks(size=18)
        plt.yticks(size=18)
        plt.xlabel('High fidelity', size=24)
        plt.ylabel('Neural network', size=24)
#        plt.xlim([-70,0])
#        plt.ylim([-70,0])
        plt.legend(['training','test'], loc=4, prop={'size':20}, frameon=False)
        plt.tight_layout()
        plt.savefig('predict_test.png')
        plt.close()
        
        raise NameError('stop')