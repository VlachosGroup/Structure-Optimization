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

import multiprocessing

import matplotlib.pyplot as plt
import matplotlib as mat
import matplotlib.ticker as mtick

from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.utils import shuffle


# Evaluation function for a given split
def eval_split(train_test_pair):
    '''
    :param train_test_pair: 3-item list of 
        1. training indices
        2. test indices
        3. NeuralNetwork object
        
    :returns: Mean square error of the fit with some holdout
    '''
    train_test_pair[2].NNModel.fit(train_test_pair[2].X[train_test_pair[0]], train_test_pair[2].Y[train_test_pair[0]])
    Y_test_pred = train_test_pair[2].NNModel.predict( train_test_pair[2].X[train_test_pair[1]] )
    return np.mean( (Y_test_pred - train_test_pair[2].Y[train_test_pair[1]]) ** 2 )

    
class NeuralNetwork():
    
    '''
    Handles training of multi-layer perceptron from sklearn
    '''
    
    def __init__(self):
        
        '''
        Initialize with empty data
        '''
        
        self.X = None           # numpy array of x values
        self.Y = None           # numpy array of y values
        self.X_scaler = None        
        self.Y_scaler = None        # Use the StandardScaler class
            
        self.Y_nn = None        # y values predicted by the neural network
        self.NNModel = None
        

    def predict(self, x):
        
        '''
        Compute value for the model as trained so far
        '''

        Ypred = self.NNModel.predict( x )   
        return Ypred
    
    
    def refine(self, X_plus, Y_plus, reg_param = 1.0):
        
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
            self.NNModel = MLPRegressor(activation = 'relu', verbose=True, learning_rate_init=0.001,
                alpha = reg_param, hidden_layer_sizes = (144,))
            self.NNModel.fit(self.X, self.Y)
        else:                   # Refining the model
            self.NNModel.partial_fit(X_plus, Y_plus)
        
        CPU_end = time.time()
        print('Neural network training time: ' + str(CPU_end - CPU_start) + ' seconds')
        
        self.Y_nn = self.predict( self.X )      # store predicted values


    def plot_parity(self, fname = 'parity.png', title = None, logscale = False, limits = None):
    
        '''
        Plot a parity plot for the data trained so far
        '''
    
        mat.rcParams['mathtext.default'] = 'regular'
        mat.rcParams['text.latex.unicode'] = 'False'
        mat.rcParams['legend.numpoints'] = 1
        mat.rcParams['lines.linewidth'] = 2
        mat.rcParams['lines.markersize'] = 12
        
        '''
        Should loop through all outputs (if there are multiple)
        '''
        
        # Figure out how many outputs there are
        if len(self.Y.shape) == 1:
            Y_d = 1
        else:
            Y_d = self.Y.shape[1]
        
        for i in range(Y_d):
        
            if Y_d == 1:
                data_1 = self.Y
                data_2 = self.Y_nn
            else:
                data_1 = self.Y[:,i]
                data_2 = self.Y_nn[:,i]
                
            plt.figure()
            plt.plot(data_1, data_2, 'o')  # Can do this for all outputs
            par_min = min( [ np.min(data_1), np.min(data_2)] )
            par_max = max( [ np.max(data_1), np.max(data_2)] )
            #plt.plot(self.Y[:,0], self.Y_nn[:,0], 'o')  # Can do this for all outputs
            #par_min = np.min( np.vstack([self.Y[:,0], self.Y_nn[:,0]]) )
            #par_max = np.max( np.vstack([self.Y[:,0], self.Y_nn[:,0]]) )
            plt.plot( [par_min, par_max], [par_min, par_max], '-', color = 'k')  # Can do this for all outputs
            
            plt.xticks(size=18)
            plt.yticks(size=18)
            plt.xlabel('High fidelity', size=24)
            plt.ylabel('Neural network', size=24)
            if not limits is None:
                plt.xlim(limits)
                plt.ylim(limits)
            if not title is None:
                plt.title('Y_' + str(i) + '_' + title, size = 24)
            #plt.legend(series_labels, loc=4, prop={'size':20}, frameon=False)
            plt.tight_layout()
            
            if logscale:
                plt.yscale('log')
                
            plt.savefig('Y_' + str(i) + '_' + fname)
            plt.close()
        
    
    def k_fold_CV(self, k = 10, reg_param = 1.0, parallel = True):

        '''
        k-fold cross validation
        
        :param k: Number of segments to split the data set. By dafault, use 10-fold cross validation
        
        :param reg_param: Regularization parameter, lambda
        
        :param test_set_size: Fraction of the data set used for validation
        
        :returns: cross-validation score
        '''
        
        pool = multiprocessing.Pool()       # For parallelization
        
        self.X, self.Y = shuffle(self.X, self.Y)

        # Neural network set up
        #N_nodes = self.X.shape[1]
        hidden_layer_sizes = (144,) # (#perceptrons in layer 1, #perceptrons in layer 2, #perceptrons in layer 3, ...)
        ## Build Model
        self.NNModel = MLPRegressor(activation = 'relu', verbose=False, learning_rate_init=0.01,
                alpha = reg_param, hidden_layer_sizes = (144,))
        
        # Populate training and testing set lists
        kf = KFold(n_splits=k)
        
        if parallel:
        
            train_test_list = []
            for train, test in kf.split(self.X, y = self.Y):
                train_test_list.append([train, test, self])
            MSEs = pool.map(eval_split, train_test_list)
            
        else:
        
            MSEs = []
            for train, test in kf.split(self.X, y = self.Y):
                self.NNModel.fit(self.X[train], self.Y[train])
                Y_test_pred = self.NNModel.predict( self.X[test] )
                MSEs.append( np.mean( (Y_test_pred - self.Y[test]) ** 2 ) )
            
        return np.sqrt( np.mean(np.array(MSEs)) )