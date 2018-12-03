# Read X and Y and train the neural network 

import numpy as np
import matplotlib as mat
import matplotlib.pyplot as plt

from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle


class surrogate(object):

    '''
    Surrogate model replacing KMC evaluation
    '''

    def __init__(self):
        
        self.cat = None
        self.predictor = None               # Neural network for predicting the activity of active sites
            
        self.all_syms = None                # All rotations and translations of training structure
        self.X_reg = None                   # Occupancies used in regression
        self.Y_reg = None                   # Site rates used in the regression
        self.site_is_active = None          # 0 for inactive sites and 1 for active sites
        
        Y_scaler = None
        
    
    def eval_structure_rate(self, all_trans, normalize = False):
    
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
            site_rates = Y_scaler.inverse_transform( self.predictor.predict( all_trans[active_site_list,:] ) )
            if normalize:
                return np.sum( site_rates ) / len(all_trans)
            else:
                return np.sum( site_rates )

                
    def eval_site_rate(self, all_trans, normalize = False):
        pass
    
    
    def train(structure_occs, site_rates, curr_fldr = None):
        '''
        Assign data and train models
        Can use some cleanup
        '''
        self.all_syms = structure_occs      
        self.partition_data_set(site_rates)
        self.train_classifier()
        self.train_regressor(reg_parity_fname = os.path.join( curr_fldr , 'site_parity'))
    
    
    def partition_data_set(self, site_rates):
        
        '''
        Split the data set into active and inactive sites
        '''
        
        # Duplicate site rates for each rotation
        site_rates_flat = site_rates.flatten()
        y = np.tile(site_rates_flat,[3,1])
        site_rates_flat = np.transpose(y).flatten()
        y_max = np.max(site_rates_flat)
        print y_max
        if y_max == 0.:
            raise NameError('No active sites in database.')
        
        self.X_reg = []
        self.Y_reg = []
        self.site_is_active = np.zeros(len(site_rates_flat))
        for index in range(len(site_rates_flat)):
            
                if site_rates_flat[index] > 0.06 * y_max:
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
        
        # Use this section for choosing hyperparameters
        self.classifier = tree.DecisionTreeClassifier(max_depth=30, random_state=0)
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
        
        
    
    def train_regressor(self, reg_parity_fname = None):
        
        '''
        Train a neural network to predict site rates
        '''

        plt.hist(self.Y_reg)
        plt.xlabel('Rate (s^-1)', size=24)
        plt.ylabel('Frequency', size=24)
        plt.xticks(size=20)
        plt.yticks(size=20)
        plt.tight_layout()
        plt.savefig('site_rate_hist')
        plt.close()

        # Scale Y
        Y_scaler = StandardScaler(with_mean = True, with_std = True)
        Y_scaler.fit(self.Y_reg.reshape(-1,1))
        Y = Y_scaler.transform(self.Y_reg.reshape(-1,1))
        Y = Y.flatten()
        
        '''
        Train neural network - Regression
        '''
        
        self.predictor = MLPRegressor(activation = 'relu', verbose=True, learning_rate_init=0.0001,
                        alpha = 0.1, hidden_layer_sizes = (20,), max_iter=10000, tol=0.00001)                
        self.predictor.fit(self.X_reg, Y)
        predictions = Y_scaler.inverse_transform( self.predictor.predict(self.X_reg) )
        mae = np.mean( np.abs( predictions - self.Y_reg ) )
        
        print 'Mean y: ' + str(np.mean(self.Y_reg))
        print 'Mean absolute error: ' + str(mae)
        
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