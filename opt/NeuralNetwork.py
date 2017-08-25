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
import time

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
    
        #for i in range(len(self.Ymeans)):
        #    Ypred[:,i] = self.Ystds[i] * Ypred[:,i] + self.Ymeans[i] * np.ones(Ypred[:,i].shape)
            
        return Ypred
    
    
    def refine(self, X_plus, Y_plus, reg_param = 0.01):
        
        '''
        Add new data to the training set and retrain
        X_plus and Y_plus need to be 2-D matrices of data
        
        :param X_plus: New x values
        
        :param Y_plus: New y values
        
        :param reg_param: Regularization parameter, lambda
        '''
        
        original = True
        # Append to existing data sets
        if self.X is None or self.Y is None:
            self.X = X_plus
            self.Y = Y_plus
        else:
            self.X = np.vstack( [ self.X , X_plus ] )
            self.Y = np.vstack( [ self.Y , Y_plus ] )
            original = False
        
        # Regress the neural network
        CPU_start = time.time()

        #self.standardize_outputs()
        #Y = self.normalized_outputs(Y = Y_plus)
        
        # Perform regression
        
        original = True         # partial fit is not giving me good results...
        if original:            # Fitting the model for the first time
            self.NNModel = MLPRegressor(activation = 'relu', verbose=True, learning_rate_init=0.01,
                alpha = reg_param, hidden_layer_sizes = (144,))
            self.NNModel.fit(self.X, self.Y)
        else:                   # Refining the model
            self.NNModel.partial_fit(X_plus, Y_plus)
        
        CPU_end = time.time()
        print('Neural network training time: ' + str(CPU_end - CPU_start) + ' seconds')
        
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
        plt.plot(self.Y, self.Y_nn, 'o')  # Can do this for all outputs
        par_min = min( [ np.min(self.Y), np.min(self.Y_nn)] )
        par_max = max( [ np.max(self.Y), np.max(self.Y_nn)] )
        #plt.plot(self.Y[:,0], self.Y_nn[:,0], 'o')  # Can do this for all outputs
        #par_min = np.min( np.vstack([self.Y[:,0], self.Y_nn[:,0]]) )
        #par_max = np.max( np.vstack([self.Y[:,0], self.Y_nn[:,0]]) )
        plt.plot( [par_min, par_max], [par_min, par_max], '-', color = 'k')  # Can do this for all outputs
        
        plt.xticks(size=18)
        plt.yticks(size=18)
        plt.xlabel('High fidelity', size=24)
        plt.ylabel('Neural network', size=24)
        #plt.xlim([1.4,2.6])
        #plt.ylim([1.4,2.6])
        if not title is None:
            plt.title(title, size = 24)
        #plt.legend(series_labels, loc=4, prop={'size':20}, frameon=False)
        plt.tight_layout()
        
        if logscale:
            plt.yscale('log')
        
        plt.savefig('Y1_' + fname)
        plt.close()
        
        #plt.figure()
        #plt.plot(self.Y[:,1], self.Y_nn[:,1], 'o')  # Can do this for all outputs
        #
        #par_min = np.min( np.vstack([self.Y[:,1], self.Y_nn[:,1]]) )
        #par_max = np.max( np.vstack([self.Y[:,1], self.Y_nn[:,1]]) )
        #plt.plot([par_min, par_max], [par_min, par_max], '-', color = 'k')
        #
        #plt.xticks(size=18)
        #plt.yticks(size=18)
        #plt.xlabel('High fidelity', size=24)
        #plt.ylabel('Neural network', size=24)
        ##plt.xlim([0,100])
        ##plt.ylim([0,100])
        #if not title is None:
        #    plt.title(title, size = 24)
        ##plt.legend(series_labels, loc=4, prop={'size':20}, frameon=False)
        #plt.tight_layout()
        #
        #if logscale:
        #    plt.yscale('log')
        #
        #plt.savefig('Y2_' + fname)
        #plt.close()
        
    
    def standardize_outputs(self):
        '''
        Normalize the y values
        '''

        self.Ymeans = []
        self.Ystds = []
        for i in range(self.Y.shape[1]):
            self.Ymeans.append( np.mean( self.Y[:,i] ) )
            self.Ystds.append( np.std( self.Y[:,i] ) )
    
    
    def normalized_outputs(self, Y = None):
        '''
        Return normalized outputs
        
        :returns: Normalized outputs
        '''
        if Y is None:
            Y = copy.deepcopy(self.Y)
        for i in range(Y.shape[1]):
            Y[:,i] = Y[:,i] - self.Ymeans[i] * np.ones(Y[:,i].shape)
            if self.Ystds[i] > 0:
                Y[:,i] = Y[:,i] / self.Ystds[i]
                
        return Y
    
    
    def train_CV(self, reg_param = 10.0, test_set_size = 0.2):

        '''
        Train the neural network with the available data
        
        :param reg_param: Regularization parameter, lambda
        :param test_set_size: Fraction of the data set used for validation
        '''
        
        # Neural network set up
        #N_nodes = self.X.shape[1]
        hidden_layer_sizes = (144,) # (#perceptrons in layer 1, #perceptrons in layer 2, #perceptrons in layer 3, ...)
        
        X = self.X
        self.standardize_outputs()
        Y = self.normalized_outputs()
        
        
        '''
        Perform regression
        '''
        
        ## Split Data
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_set_size, random_state=1)
            
        ## Build Model
        self.NNModel = MLPRegressor(activation = 'relu', solver = 'lbfgs', alpha = reg_param, hidden_layer_sizes = hidden_layer_sizes)
        
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