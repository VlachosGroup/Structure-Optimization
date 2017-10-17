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

    
class NeuralNetwork(MLPRegressor):
    
    '''
    Adds a little functionality onto the multi-layer perceptron of sklearn
    1. Stores the data used to train
    2. Parallelized CV score
    3. Plotting of parity plots for each objective
    '''
    
    def __init__(self, activation = 'relu', verbose=True, learning_rate_init=0.0001,
                alpha = 1.0, hidden_layer_sizes = (81,), max_iter=10000, tol=0.00001):
        
        '''
        Initialize the neural network with default parameters and empty data
        '''
        
        MLPRegressor.__init__(self, activation = activation, verbose=verbose, learning_rate_init=learning_rate_init,
                alpha = alpha, hidden_layer_sizes = hidden_layer_sizes, max_iter=max_iter, tol=tol)
        
        self.X = None           # numpy array of x values
        self.Y = None           # numpy array of y values
        self.Y_norm = 1
        
        self.X_scaler = None        
        self.Y_scaler = None        # Use the StandardScaler class
            
        self.Y_nn = None        # y values predicted by the neural network
    
    
    def refine(self, X_plus, Y_plus, weight_active = True):
        
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
            if len(self.Y.shape) == 1:      # If y is a 1-D array
                self.Y = np.hstack( [ self.Y , Y_plus ] )
            else:
                self.Y = np.vstack( [ self.Y , Y_plus ] )
            original = False
        
        self.Y_norm = np.max(self.Y)
        
        '''
        Weight the most active sites
        '''
        
        if weight_active:
        
            active_cutoff = 0.07
            target_active = 0.5
            
            n_data = len(self.Y)
            n_inactive = 0
            for i in range(n_data):
                if self.Y[i] / self.Y_norm < active_cutoff:
                    n_inactive += 1
            n_active = n_data - n_inactive
            
                    
            weight = int( target_active * n_inactive / n_active / ( 1 - target_active ) ) 
            
            # Duplicate active sites
            for i in range(n_data):
                if self.Y[i] / self.Y_norm >= active_cutoff:
                    for j in xrange(weight):
                        self.X = np.vstack([self.X, self.X[i,:]])
                        self.Y = np.hstack([self.Y, np.array(self.Y[i])])
        
        # Regress the neural network
        CPU_start = time.time()
        
        # Perform regression
        
        original = True         # partial fit is not giving me good results...
        if original:            # Fitting the model for the first time
            self.fit(self.X, self.Y / self.Y_norm)
        else:                   # Refining the model
            self.partial_fit(X_plus, Y_plus)
        
        CPU_end = time.time()
        print('Neural network training time: ' + str(CPU_end - CPU_start) + ' seconds')
        
        self.Y_nn = self.Y_norm * self.predict( self.X )      # store predicted values


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
                plt.title('Y_' + str(i+1) + '_' + title, size = 24)
            #plt.legend(series_labels, loc=4, prop={'size':20}, frameon=False)
            plt.tight_layout()
            
            if logscale:
                plt.yscale('log')
            
            if Y_d > 1:            
                plt.savefig('Y_' + str(i+1) + '_' + fname)
            else:
                plt.savefig(fname)
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
        
        ## Build Model
        self = MLPRegressor(activation = 'relu', verbose=False, learning_rate_init=0.01,
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
                self.fit(self.X[train], self.Y[train])
                Y_test_pred = self.predict( self.X[test] )
                MSEs.append( np.mean( (Y_test_pred - self.Y[test]) ** 2 ) )
            
        return np.sqrt( np.mean(np.array(MSEs)) )