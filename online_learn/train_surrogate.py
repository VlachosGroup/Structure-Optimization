# Read X and Y and train the neural network 

import numpy as np
import matplotlib as mat
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from NH3.NiPt_NH3_simple import NiPt_NH3_simple
from sklearn.model_selection import train_test_split
import pickle

from optimize_structure import eval_rate


def eval_rate(all_trans, nn_class, nn_pred, normalize_fac = 144):

    '''
    Surrogate model using a classifier and regressor
    '''
    
    if all_trans is None:
        cat.variable_occs = sigma
        all_trans = cat.generate_all_translations()
    sites_are_active = nn_class.predict(all_trans)
    
    active_site_list = []
    for site_ind in range(len(sites_are_active)):
        if sites_are_active[site_ind] == 1.:
            active_site_list.append(site_ind)
    
    if active_site_list == []:
        return 0
    else:
        return np.sum( nn_pred.predict( all_trans[active_site_list,:] ) ) / normalize_fac


def generate_symmetries(structure_occs):

    '''
    :param structure_occs:   Occupancy vectors for each structure
    :param site_propensities:   Site propensities for each structure
    '''

    cat = NiPt_NH3_simple()
    all_syms = None
    
    for x in structure_occs:
    
        cat.variable_occs = x
        #all_trans = cat.generate_all_translations()
        all_trans = cat.generate_all_translations_and_rotations()
        if all_syms is None:
            all_syms = all_trans
        else:
            all_syms = np.vstack([all_syms, all_trans])
        
    return all_syms
    
        
def partition_data_set(all_syms, site_rates):
    
    '''
    Split the data set into active and inactive sites
    '''
    
    # Duplicate data points for each rotation
    site_rates_flat = site_rates.flatten()
    y = np.tile(site_rates_flat,[3,1])
    site_rates_flat = np.transpose(y).flatten()
    y_max = np.max(site_rates_flat)
    active_cut = 0.001
    
    X_reg = []
    Y_reg = []
    site_is_active = np.zeros(len(site_rates_flat))
    index = 0
    for index in range(len(site_rates_flat)):
        
            if site_rates_flat[index] > active_cut * y_max :
                X_reg.append( all_syms[index,:] )
                Y_reg.append( site_rates_flat[index] )
                site_is_active = np.zeros(len(site_rates_flat)) = 1     # Classify rates as zero or nonzero

    X_reg = np.array(X_reg)
    Y_reg = np.array(Y_reg)
    
    return [site_is_active, X_reg, Y_reg]
    
    

def train_classifier(all_syms, site_is_active, pickle_name = None):    
    
    '''
    Train neural network - Classification
    '''
    
    attempts = 0
    good_enough = False
    
    while not good_enough:
    
        if attempts > 10:
            raise NameError('Classification unable to converge in ' + str(attempts) + ' attempts.')
    
        classifier = MLPClassifier(activation = 'relu', verbose=True, learning_rate_init=0.01,
                        alpha = 0.1, hidden_layer_sizes = (50,))                
        classifier.fit(all_syms, site_is_active)
        predictoriction_train = classifier.predict(all_syms)
        frac_wrong_train = np.float( np.sum( np.abs( predictoriction_train - site_is_active ) ) ) / len(site_is_active)
        print 'Fraction wrong in training set: ' + str(frac_wrong_train)
        
        good_enough = frac_wrong_train < 0.01        # Need classification to be at least 99% accurate
        good_enough = True
        attempts += 1
    
    if not pickle_name is None:
        pickle.dump( classifier, open( pickle_name, 'wb' ) )
    
    return classifier


def train_regressor(X_reg, Y_reg, pickle_name = None):    
    
    '''
    Train neural network - Regression
    '''
    
    attempts = 0
    good_enough = False
    
    while not good_enough:
    
        if attempts > 10:
            raise NameError('Neural network Regression unable to converge in 11 attempts.')
    
        if attempts > 0:
            print 'Retrying neural network regression, attempt # ' + str(attempts+1)
        
        
        predictor = MLPRegressor(activation = 'relu', verbose=True, learning_rate_init=0.0001,
                        alpha = 1.0, hidden_layer_sizes = (50,), max_iter=10000, tol=0.0001)                
        predictor.fit(X_reg, Y_reg)
        predictoriction = predictor.predict(X_reg)
        mae = np.mean( np.abs( predictoriction - Y_reg ) )
        
        print 'Mean y: ' + str(np.mean(Y_reg))
        print 'Mean absolute error: ' + str(mae)
        
        good_enough = mae / np.mean(Y_reg) < 0.5       # error in quantitative predictions must be within 25% of the mean value
        good_enough = True
        attempts += 1
    
    return predictor
    
    
    '''
    Parity plot for the neural network fit
    '''
    
    mat.rcParams['mathtext.default'] = 'regular'
    mat.rcParams['text.latex.unicode'] = 'False'
    mat.rcParams['legend.numpoints'] = 1
    mat.rcParams['lines.linewidth'] = 2
    mat.rcParams['lines.markersize'] = 12
    
    plt.figure()
    plt.plot(Y_reg, predictoriction, 'o')
    all_point_values = np.hstack([Y_reg, predictoriction])
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
    plt.savefig('Iteration_' + str(iter_num) + '_training_site_parity_rot', dpi = 600)
    plt.close()
    
    if not pickle_name is None:
        pickle.dump( predictor, open( pickle_name, 'wb' ) )
    
    raise NameError('stop')
    
    
 