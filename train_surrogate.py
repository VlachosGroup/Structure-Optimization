# Read X and Y and train the neural network 

import numpy as np
import matplotlib as mat
import matplotlib.pyplot as plt

from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from NH3.NiPt_NH3_simple import NiPt_NH3_simple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle


class surrogate(object):

    '''
    Surrogate model replacing KMC evaluation
    '''

    def __init__(self):
        
        self.classifier = None              # Decision tree for classifying sites as active or inactive
        self.predictor = None               # Neural network for predicting the activity of active sites
            
        self.all_syms = None                # All rotations and translations of training structure
        self.X_reg = None                   # Occupancies used in regression
        self.Y_reg = None                   # Site rates used in the regression
        self.site_is_active = None          # 0 for inactive sites and 1 for active sites
        
        self.Y_scaler = None
        
    
    def eval_rate(self, all_trans, normalize = False):
    
        '''
        Surrogate model using a classifier and regressor
        '''
        
        sites_are_active = self.classifier.predict(all_trans)
        
        active_site_list = []
        for site_ind in range(len(sites_are_active)):
            if sites_are_active[site_ind] == 1.:
                active_site_list.append(site_ind)
        
        if active_site_list == []:
            return 0
        else:
            site_rates = self.Y_scaler.inverse_transform( self.predictor.predict( all_trans[active_site_list,:] ) )
            if normalize:
                return np.sum( site_rates ) / len(site_rates)
            else:
                return np.sum( site_rates )

    
    def train_decision_tree_regressor(self,site_rates):
        '''
        Try doing all regression with decision tree instead
        '''
        
        
        plt.hist(self.Y_reg)
        plt.xlabel('Rate (s^-1)', size=24)
        plt.ylabel('Frequency', size=24)
        plt.xticks(size=20)
        plt.yticks(size=20)
        plt.tight_layout()
        plt.savefig('site_rate_hist')
        plt.close()
        raise NameError('stop')
        # Duplicate site rates for each rotation
        #site_rates_flat = site_rates.flatten()
        #y = np.tile(site_rates_flat,[3,1])
        #site_rates_flat = np.transpose(y).flatten()
        #y_max = np.max(site_rates_flat)
        #
        #X = self.all_syms
        #Y = site_rates_flat / y_max
        
        y_max = np.max(self.Y_reg)
        X = self.X_reg
        Y = self.Y_reg / y_max
        
        dtr = tree.DecisionTreeRegressor(max_depth=15)#(max_leaf_nodes=150)
        X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=0)
        dtr.fit(X_train, y_train)
        predictions_train = y_max * dtr.predict(X_train)
        predictions_test = y_max * dtr.predict(X_test)
        y_train_raw = y_max * y_train
        y_test_raw = y_max * y_test
        
        # Print information
        print dtr.tree_.node_count
        print dtr.tree_.children_left
        print dtr.tree_.children_right
        print dtr.tree_.feature
        print dtr.tree_.threshold
        
        '''
        Plot parity
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
        plt.plot( [par_min, par_max], [par_min, par_max], '--', color = 'k')
        plt.plot(y_train_raw, predictions_train, 'o', color = 'b', label = 'train')
        plt.plot(y_test_raw, predictions_test, 'o', color = 'r', label = 'test')
        
        plt.xticks(size=18)
        plt.yticks(size=18)
        plt.xlabel('Kinetic Monte Carlo', size=24)
        plt.ylabel('Neural network', size=24)
        plt.legend(loc=4, prop={'size':20}, frameon=False)
        plt.tight_layout()
        plt.savefig('decision_tree_fit', dpi = 600)
        plt.close()
        
        raise NameError('stop')
    
    def partition_data_set(self,site_rates):
        
        '''
        Split the data set into active and inactive sites
        '''
        
        # Duplicate site rates for each rotation
        site_rates_flat = site_rates.flatten()
        y = np.tile(site_rates_flat,[3,1])
        site_rates_flat = np.transpose(y).flatten()
        y_max = np.max(site_rates_flat)
        active_cut = 0.005
        
        if y_max == 0.:
            raise NameError('No active sites in database.')
        
        self.X_reg = []
        self.Y_reg = []
        self.site_is_active = np.zeros(len(site_rates_flat))
        for index in range(len(site_rates_flat)):
            
                if site_rates_flat[index] > active_cut * y_max :
                    self.X_reg.append( self.all_syms[index,:] )
                    self.Y_reg.append( site_rates_flat[index] )
                    self.site_is_active[index] = 1     # Classify rates as zero or nonzero
    
        self.X_reg = np.array(self.X_reg)
        self.Y_reg = np.array(self.Y_reg)
        
    
    def train_classifier(self):    
        
        '''
        Train decision tree
        
        :param self.all_syms: All permutations of input structures
        :param self.site_is_active: 0 if inactive, 1 if active
        '''
        
        self.classifier = tree.DecisionTreeClassifier(max_leaf_nodes=50)
        self.classifier.fit(self.all_syms, self.site_is_active)
        predictoriction_train = self.classifier.predict(self.all_syms)
        frac_wrong_train = np.float( np.sum( np.abs( predictoriction_train - self.site_is_active ) ) ) / len(self.site_is_active)
        print 'Fraction wrong in training set: ' + str(frac_wrong_train)
        '''
        # Use this section for choosing hyperparameters
        self.classifier = tree.DecisionTreeClassifier(max_leaf_nodes=70, random_state=0)
        #self.classifier = tree.DecisionTreeClassifier(random_state=0)
        X_train, X_test, y_train, y_test = train_test_split(self.all_syms, self.site_is_active, random_state=0)
        self.classifier.fit(X_train, y_train)
        predictoriction_train = self.classifier.predict(X_train)
        frac_wrong_train = np.float( np.sum( np.abs( predictoriction_train - y_train ) ) ) / len(y_train)
        print 'Fraction wrong in training set: ' + str(frac_wrong_train)
        predictoriction_test = self.classifier.predict(X_test)
        frac_wrong_test = np.float( np.sum( np.abs( predictoriction_test - y_test ) ) ) / len(y_test)
        print 'Fraction wrong in test set: ' + str(frac_wrong_test)
        print 'Fraction in training set: ' + str(np.float(len(y_train)) / (len(y_test) + len(y_train)))
        
        n_nodes = self.classifier.tree_.node_count
        children_left = self.classifier.tree_.children_left
        children_right = self.classifier.tree_.children_right
        feature = self.classifier.tree_.feature
        threshold = self.classifier.tree_.threshold
        
        print n_nodes
        print children_left
        print children_right
        print feature
        print threshold
        raise NameError('stop')
        '''
    
    def train_regressor(self, reg_parity_fname = None):
        
        '''
        Train a neural network to predict site rates
        '''

        # Scale Y
        self.Y_scaler = StandardScaler(with_mean = True, with_std = True)
        self.Y_scaler.fit(self.Y_reg.reshape(-1,1))
        Y = self.Y_scaler.transform(self.Y_reg.reshape(-1,1))
        Y = Y.reshape(-1,1)
        
        '''
        Train neural network - Regression
        '''
        
        #self.predictor = MLPRegressor(activation = 'relu', verbose=True, learning_rate_init=0.0001,
        #                alpha = 0.1, hidden_layer_sizes = (20,), max_iter=10000, tol=0.00001)                
        #self.predictor.fit(self.X_reg, Y)
        #predictions = self.Y_scaler.inverse_transform( self.predictor.predict(self.X_reg) )
        #mae = np.mean( np.abs( predictions - self.Y_reg ) )
        #
        #print 'Mean y: ' + str(np.mean(self.Y_reg))
        #print 'Mean absolute error: ' + str(mae)
        
        # Use this section for choosing hyperparameters
        
        self.predictor = MLPRegressor(activation = 'relu', verbose=True, learning_rate_init=0.0001,
                        alpha = 1.0, hidden_layer_sizes = (30,), max_iter=10000, tol=0.00001)
        X_train, X_test, y_train, y_test = train_test_split(self.X_reg, Y, random_state=0)
        y_train_raw = self.Y_scaler.inverse_transform(y_train)
        y_test_raw = self.Y_scaler.inverse_transform(y_test)
        self.predictor.fit(X_train, y_train)
        print 'Mean y: ' + str(np.mean(self.Y_reg))
        predictions_train = self.Y_scaler.inverse_transform( self.predictor.predict(X_train) )
        predictions_train = predictions_train.reshape(-1)
        y_train_raw = y_train_raw.reshape(-1)
        mae = np.mean( np.abs( predictions_train - y_train_raw ) )
        print 'Mean absolute error (train): ' + str(mae)
        predictions_test = self.Y_scaler.inverse_transform( self.predictor.predict(X_test) )
        predictions_test = predictions_test.reshape(-1)
        y_test_raw = y_test_raw.reshape(-1)
        mae = np.mean( np.abs( predictions_test - y_test_raw ) )
        print 'Mean absolute error (test): ' + str(mae)
        
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
        plt.plot(y_test_raw, predictions_test, 'o', color = 'r', label = 'test')
        
        plt.xticks(size=18)
        plt.yticks(size=18)
        plt.xlabel('Kinetic Monte Carlo', size=24)
        plt.ylabel('Neural network', size=24)
        plt.legend(loc=4, prop={'size':20}, frameon=False)
        plt.tight_layout()
        plt.savefig('hyper_fit', dpi = 600)
        plt.close()
        
        raise NameError('stop')
        
        
        '''
        Parity plot for the neural network fit
        '''
        
        if not reg_parity_fname is None:
        
            mat.rcParams['mathtext.default'] = 'regular'
            mat.rcParams['text.latex.unicode'] = 'False'
            mat.rcParams['legend.numpoints'] = 1
            mat.rcParams['lines.linewidth'] = 2
            mat.rcParams['lines.markersize'] = 12
            
            plt.figure()
            all_point_values = np.hstack([self.Y_reg, predictions])
            par_min = min( all_point_values )
            par_max = max( all_point_values )
            plt.plot( [par_min, par_max], [par_min, par_max], '--', color = 'k')  # Can do this for all outputs
            plt.plot(self.Y_reg, predictions, 'o')
            
            plt.xticks(size=18)
            plt.yticks(size=18)
            plt.xlabel('Kinetic Monte Carlo', size=24)
            plt.ylabel('Neural network', size=24)
            plt.tight_layout()
            plt.savefig(reg_parity_fname, dpi = 600)
            plt.close()