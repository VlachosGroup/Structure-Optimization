# Read X and Y and train the neural network 

import numpy as np
import matplotlib as mat
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from NH3.NiPt_NH3_simple import NiPt_NH3_simple
from sklearn.model_selection import train_test_split
import pickle



def train_neural_nets(structure_occs, site_propensities, iter_num = None):

    '''
    :param structure_occs:   Occupancy vectors for each structure
    :param site_propensities:   Site propensities for each structure
    '''

    n_sims = site_propensities.shape[0]
    n_sites = site_propensities.shape[1]
    n_rxns = site_propensities.shape[2]
    
    site_rates = site_propensities[:,:,0] - site_propensities[:,:,1]        # A adsorption
    
    '''
    Create X
    '''
    
    cat = NiPt_NH3_simple()
    all_x = None
    
    for x in structure_occs:
    
        cat.variable_occs = x
        all_trans = cat.generate_all_translations()
        if all_x is None:
            all_x = all_trans
        else:
            all_x = np.vstack([all_x, all_trans])
        
    n_data = all_x.shape[0]
        
    '''
    Create Y
    '''
    
    X_reg = []
    Y_reg = []
    index = 0
    for i in range(n_sims):
        for j in range(n_sites):
        
            if site_rates[i,j] > 0. :               # possibly split this into zero and nonzero, since rates can be negative
                X_reg.append( all_x[index,:] )
                Y_reg.append( site_rates[i,j] )
                
            index += 1
    
    site_rates_flat = site_rates.flatten()
    
    
    # Classify rates as zero or nonzero
    site_is_active = np.zeros(len(site_rates_flat))
    
    for i in range(n_data):
        if site_rates_flat[i] > 0. :
            site_is_active[i] = 1
    
    '''
    Reduce to local indices (if necessary)
    '''
    
    # Remove elements of X that are far from the active site being evaluated
    # Copy from previous script
    
    
    '''
    Train neural network - Classification
    '''
    
    X_train, X_test, Y_train, Y_test = train_test_split(all_x, site_is_active, test_size=0.1, random_state=1)
    
    nn_class = MLPClassifier(activation = 'relu', verbose=True, learning_rate_init=0.01,
                    alpha = 0.1, hidden_layer_sizes = (50,))                
    nn_class.fit(X_train, Y_train)
    nn_prediction_train = nn_class.predict(X_train)
    nn_prediction_test = nn_class.predict(X_test)
    frac_wrong_train = np.float( np.sum( np.abs( nn_prediction_train - Y_train ) ) ) / len(Y_train)
    frac_wrong_test = np.float( np.sum( np.abs( nn_prediction_test - Y_test ) ) ) / len(Y_test)
    print 'Fraction wrong in training set: ' + str(frac_wrong_train)
    print 'Fraction wrong in test set: ' + str(frac_wrong_test)
    
    # Refine with the test data
    nn_class.partial_fit(X_test, Y_test)
    
    '''
    Train neural network - Regression
    '''
    
    X_train, X_test, Y_train, Y_test = train_test_split(X_reg, Y_reg, test_size=0.1, random_state=1)
    
    
    nn_pred = MLPRegressor(activation = 'relu', verbose=True, learning_rate_init=0.001,
                    alpha = 0.1, hidden_layer_sizes = (50,), max_iter=10000, tol=0.0001)                
    nn_pred.fit(X_train, Y_train)
    nn_prediction_trained = nn_pred.predict(X_train)
    nn_prediction_test = nn_pred.predict(X_test)
    
    # Refine with the test data
    nn_pred.partial_fit(X_test, Y_test)
    
    '''
    Parity plot for the neural network fit
    '''
    
    mat.rcParams['mathtext.default'] = 'regular'
    mat.rcParams['text.latex.unicode'] = 'False'
    mat.rcParams['legend.numpoints'] = 1
    mat.rcParams['lines.linewidth'] = 2
    mat.rcParams['lines.markersize'] = 12
    
    plt.figure()
    plt.plot(Y_train, nn_prediction_trained, 'o')
    plt.plot(Y_test, nn_prediction_test, '^')
    all_point_values = np.hstack([Y_train, nn_prediction_trained, Y_test, nn_prediction_test])
    par_min = min( all_point_values )
    par_max = max( all_point_values )
    plt.plot( [par_min, par_max], [par_min, par_max], '-', color = 'k')  # Can do this for all outputs
    
    plt.xticks(size=18)
    plt.yticks(size=18)
    plt.xlabel('Kinetic Monte Carlo', size=24)
    plt.ylabel('Neural network', size=24)
    #plt.xlim([0, 0.50])
    #plt.ylim([0, 0.50])
    plt.legend(['train', 'test'], loc=4, prop={'size':20}, frameon=False)
    plt.tight_layout()
    plt.savefig('Iteration_' + str(iter_num) + '_site_parity', dpi = 600)
    plt.close()
    
    
    '''
    See how the neural network predicts the overall activity of the structures in the training set
    '''
    
    structure_rates_KMC = np.sum(site_rates, axis = 1)
    structure_rates_NN = np.zeros(structure_rates_KMC.shape)
    
    cat = NiPt_NH3_simple()
    struc_ind = 0
    for x in structure_occs:
    
        cat.variable_occs = x
        all_trans = cat.generate_all_translations()
        
        sites_are_active = nn_class.predict(all_trans)
        
        active_site_list = []
        for site_ind in range(len(sites_are_active)):
            if sites_are_active[site_ind] == 1.:
                active_site_list.append(site_ind)
        
        if active_site_list == []:
            structure_rates_NN[struc_ind] = 0
        else:
            structure_rates_NN[struc_ind] = np.sum( nn_pred.predict( all_trans[active_site_list,:] ) )
        
        struc_ind += 1
    
    # Take x
    # x -> all permutations
    # is each site active? If not, activity = 0
    # If it is active, compute the activity
    # sum all site activities
    
    # Do this with both elementary steps and compare?
    
    plt.figure()
    plt.plot(structure_rates_KMC, structure_rates_NN, 'o')
    all_point_values = np.hstack([structure_rates_KMC, structure_rates_NN])
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
    plt.savefig('Iteration_' + str(iter_num) + '_structure_parity', dpi = 600)
    plt.close()
    
    pickle.dump( nn_class, open( 'Iteration_' + str(iter_num) + '_nn_class.p', 'wb' ) )
    pickle.dump( nn_pred, open( 'Iteration_' + str(iter_num) + '_nn_reg.p', 'wb' ) )
    
    return [nn_class, nn_pred]