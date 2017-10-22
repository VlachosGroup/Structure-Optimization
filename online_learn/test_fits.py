'''
Driver for the online learning program
'''

import sys
sys.path.append('/home/vlachos/mpnunez/Github/Zacros-Wrapper')
sys.path.append('/home/vlachos/mpnunez/Github/Structure-Optimization')
sys.path.append('/home/vlachos/mpnunez/python_packages/ase')
sys.path.append('/home/vlachos/mpnunez/python_packages/sklearn/lib/python2.7/site-packages')

import pickle
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
import time

from multiprocessing import Pool
from NH3.NiPt_NH3_simple import NiPt_NH3_simple
import zacros_wrapper as zw

import matplotlib as mat
import matplotlib.pyplot as plt

# functions
from read_KMC_site_props import *
from train_neural_nets import *
from optimize_structure import *
from generate_more_structures import *

if __name__ == '__main__':

    initial_DB_size = 97
    data_fldr = '/home/vlachos/mpnunez/NN_data/AB_data'
    DB_fldr = './KMC_DB'        # relative to data_fldr
    
    os.chdir(data_fldr)
    
    '''
    Read KMC data -> build X and Y
    '''
    
    # Read from existing pickle file
    if os.path.isfile('Iteration_0_X.npy'): 
    
        structure_occs = np.load('Iteration_0_X.npy')
        site_propensities = np.load('Iteration_0_Y.npy')
    
    # Read all KMC files
    else:
    
        fldr_list = [os.path.join(DB_fldr, 'structure_' + str(i+1) ) for i in xrange(initial_DB_size)]
    
        # Run in parallel
        pool = Pool()
        x_vec = pool.map(read_x, fldr_list)
        pool.close()
        
        pool = Pool()
        y_vec = pool.map(read_y, fldr_list)
        pool.close()
        
        structure_occs = np.array(x_vec)
        site_propensities = np.array(y_vec)
        
        np.save('Iteration_0_X.npy', structure_occs)
        np.save('Iteration_0_Y.npy', site_propensities)
    
    
    '''
    X and Y -> neural network objects
    '''

    '''
    :param structure_occs:   Occupancy vectors for each structure
    :param site_propensities:   Site propensities for each structure
    '''

    n_sims = site_propensities.shape[0]
    n_sites = site_propensities.shape[1]
    n_rxns = site_propensities.shape[2]
    
    site_rates = site_propensities[:,:,0] - site_propensities[:,:,1]        # A adsorption
    #site_rates = site_propensities[:,:,-2] - site_propensities[:,:,-1]        # A adsorption
    
    '''
    Create X
    '''
    CPU_start = time.time()
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
    X_reg = np.array(X_reg)
    Y_reg = np.array(Y_reg)
    
    # Classify rates as zero or nonzero
    site_is_active = np.zeros(len(site_rates_flat))
    
    for i in range(n_data):
        if site_rates_flat[i] > 0. :
            site_is_active[i] = 1
    
    
    
    #print site_rates_flat[0:144:]
    #print all_x[7,:]
    #raise NameError('stop')
    
    '''
    Reduce to local indices (if necessary)
    '''
    
    # Remove elements of X that are far from the active site being evaluated
    # Copy from previous script
    
    
    '''
    Train neural network - Classification
    '''
    
    attempts = 0
    good_enough = False
    
    while not good_enough:
    
        if attempts > 10:
            raise NameError('Classification unable to converge in ' + str(attempts) + ' attempts.')
    
        nn_class = MLPClassifier(activation = 'relu', verbose=False, learning_rate_init=0.01,
                        alpha = 1., hidden_layer_sizes = (50,))                
        nn_class.fit(all_x, site_is_active)
        nn_prediction_train = nn_class.predict(all_x)
        frac_wrong_train = np.float( np.sum( np.abs( nn_prediction_train - site_is_active ) ) ) / len(site_is_active)
        print 'Fraction wrong in training set: ' + str(frac_wrong_train)
        
        good_enough = frac_wrong_train < 0.01        # Need classification to be at least 99% accurate
        good_enough = True
        attempts += 1
    
    
    '''
    Train neural network - Regression
    '''
    
    attempts = 0
    good_enough = False
    
    print str(n_sims * n_sites) + ' total sites'
    
    print str(len(Y_reg)) + ' active sites'
    
    #print Y_reg[0:144:]
    #print '\n'
    #
    #for i in range(144):
    #    print X_reg[i,:]
    #raise NameError('stop')
    X_reg = X_reg[:, cat.get_local_inds() ]
    while not good_enough:
    
        if attempts > 10:
            raise NameError('Neural network Regression unable to converge in 11 attempts.')
    
        if attempts > 0:
            print 'Retrying neural network regression, attempt # ' + str(attempts+1)
        
        
        nn_pred = MLPRegressor(activation = 'relu', verbose=True, learning_rate_init=0.001,
                    alpha = 0.1, hidden_layer_sizes = (50,), max_iter=10000, tol=0.0001)                
        nn_pred.fit(X_reg, Y_reg)
        nn_prediction = nn_pred.predict(X_reg)
        mae = np.mean( np.abs( nn_prediction - Y_reg ) )
        
        print 'Mean y: ' + str(np.mean(Y_reg))
        print 'Mean absolute error: ' + str(mae)
        
        good_enough = mae / np.mean(Y_reg) < 0.5       # error in quantitative predictions must be within 25% of the mean value
        good_enough = True
        attempts += 1
    
    CPU_end = time.time()
    print('Simulated annealing time elapsed: ' + str(CPU_end - CPU_start) )    
    raise NameError('stop')
    
    '''
    Parity plot for the neural network fit
    '''
    iter_num = 1
    mat.rcParams['mathtext.default'] = 'regular'
    mat.rcParams['text.latex.unicode'] = 'False'
    mat.rcParams['legend.numpoints'] = 1
    mat.rcParams['lines.linewidth'] = 2
    mat.rcParams['lines.markersize'] = 12
    
    plt.figure()
    plt.plot(Y_reg, nn_prediction, 'o')
    all_point_values = np.hstack([Y_reg, nn_prediction])
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
    plt.savefig('Iteration_' + str(iter_num) + '_training_site_parity', dpi = 600)
    plt.close()
    
    
    '''
    See how the neural network predicts the overall activity of the structures in the training set
    '''
    
    structure_rates_KMC = np.sum(site_rates, axis = 1)
    structure_rates_NN = np.zeros(structure_rates_KMC.shape)
    
    #cat = NiPt_NH3_simple()
    #struc_ind = 0
    #for x in structure_occs:
    #    structure_rates_NN[struc_ind] = eval_rate( cat, x, nn_class, nn_pred )
    #    struc_ind += 1
    
    
    
    
    # Take x
    # x -> all permutations
    # is each site active? If not, activity = 0
    # If it is active, compute the activity
    # sum all site activities
    
    # Do this with both elementary steps and compare?
    
    #plt.figure()
    #plt.plot(structure_rates_KMC, structure_rates_NN, 'o')
    #all_point_values = np.hstack([structure_rates_KMC, structure_rates_NN])
    #par_min = min( all_point_values )
    #par_max = max( all_point_values )
    #plt.plot( [par_min, par_max], [par_min, par_max], '-', color = 'k')  # Can do this for all outputs
    #
    #plt.xticks(size=18)
    #plt.yticks(size=18)
    #plt.xlabel('Kinetic Monte Carlo', size=24)
    #plt.ylabel('Neural network', size=24)
    ##plt.xlim([0, 0.50])
    ##plt.ylim([0, 0.50])
    ##plt.legend(['train', 'test'], loc=4, prop={'size':20}, frameon=False)
    #plt.tight_layout()
    #plt.savefig('Iteration_' + str(iter_num) + '_structure_parity', dpi = 600)
    #plt.close()