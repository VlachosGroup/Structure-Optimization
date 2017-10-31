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
        
    
    def eval_rate(self, all_trans, normalize_fac = 144):
    
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
            return np.sum( site_rates ) / normalize_fac
    
    
    def generate_symmetries(self,structure_occs):
    
        '''
        :param structure_occs:   Occupancy vectors for each structure
        :param site_propensities:   Site propensities for each structure
        '''
    
        cat = NiPt_NH3_simple()
        
        
        
        for x in structure_occs:            # Can we parallelize this?
        
            cat.variable_occs = x
            all_trans_and_rot = cat.generate_all_translations_and_rotations()
            if self.all_syms is None:
                self.all_syms = all_trans_and_rot
            else:
                self.all_syms = np.vstack([self.all_syms, all_trans_and_rot])

                
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
        
        print site_rates_flat.shape
        
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
        
        self.classifier = tree.DecisionTreeClassifier()
        self.classifier.fit(self.all_syms, self.site_is_active)
        predictoriction_train = self.classifier.predict(self.all_syms)
        frac_wrong_train = np.float( np.sum( np.abs( predictoriction_train - self.site_is_active ) ) ) / len(self.site_is_active)
        print 'Fraction wrong in training set: ' + str(frac_wrong_train)
    
    
    def train_regressor(self):    
        
        '''
        Train a neural network to predict site rates
        '''

        # Scale Y
        self.Y_scaler = StandardScaler(with_mean = True, with_std = True)
        self.Y_scaler.fit(self.Y_reg.reshape(-1,1))
        Y = self.Y_scaler.transform(self.Y_reg.reshape(-1,1))
        Y = Y.reshape(-1,1)
        
        print '\n'
        print self.Y_reg.shape
        '''
        Train neural network - Regression
        '''
        
        self.predictor = MLPRegressor(activation = 'relu', verbose=True, learning_rate_init=0.0001,
                        alpha = 0.1, hidden_layer_sizes = (20,), max_iter=10000, tol=0.00001)                
        self.predictor.fit(self.X_reg, Y)
        predictions = self.Y_scaler.inverse_transform( self.predictor.predict(self.X_reg) )
        mae = np.mean( np.abs( predictions - self.Y_reg ) )
        
        print 'Mean y: ' + str(np.mean(self.Y_reg))
        print 'Mean absolute error: ' + str(mae)
        
        
        '''
        Parity plot for the neural network fit
        '''
        
        mat.rcParams['mathtext.default'] = 'regular'
        mat.rcParams['text.latex.unicode'] = 'False'
        mat.rcParams['legend.numpoints'] = 1
        mat.rcParams['lines.linewidth'] = 2
        mat.rcParams['lines.markersize'] = 12
        
        plt.figure()
        plt.plot(self.Y_reg, predictions, 'o')
        all_point_values = np.hstack([self.Y_reg, predictions])
        par_min = min( all_point_values )
        par_max = max( all_point_values )
        plt.plot( [par_min, par_max], [par_min, par_max], '-', color = 'k')  # Can do this for all outputs
        
        plt.xticks(size=18)
        plt.yticks(size=18)
        plt.xlabel('Kinetic Monte Carlo', size=24)
        plt.ylabel('Neural network', size=24)
        #plt.xlim([0, 0.50])
        #plt.ylim([0, 0.50])
        #plt.legend(['train', 'test'], loc=4, prop={'size':20}, frameon=False)
        plt.tight_layout()
        plt.savefig('Iteration_' + str(0) + '_training_site_parity_rot', dpi = 600)
        plt.close()