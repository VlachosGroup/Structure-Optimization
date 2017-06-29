# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 14:03:35 2017

@author: mpnun
"""

import numpy as np
import random
import os

import matplotlib.pyplot as plt

from NeuralNetwork import NeuralNetwork

class Bifidelity():
    
    '''
    Genetic algorithm optimization with a high fidelity and low fidelity models
    '''
    
    def __init__(self, pop_size = None):
        
        '''
        Initialize with an empty population and no training data
        '''
        
        self.pop = None                     # List of individuals
        self.fitnesses = None               # Fitness values for the individuals in the population
        self.training_X = None              # Set of training structures used to regress the low fidelity model
        self.training_Y = None              # Training output
        self.low_fid = None                 # Low fidelity model trained with data

        if not pop_size is None:
            self.initialize_pop( pop_size )


    def eval_OF(self, x, fidelity = 'high'):
        
        '''
        Evaluate the objective function for an individual by calling either the high or low fidelity model
        '''
        
        if fidelity == 'high':
    
            '''
            Ackley function
            x is an n-dimensional vector
            '''
            
            y = np.array(x)
            return -20 * np.exp( -0.2 * np.sqrt( np.mean( y ** 2 ) ) ) - np.exp( np.mean( np.cos( 2 * np.pi * y ) ) ) + 20 + np.e


        elif fidelity == 'low':

            return self.low_fid.predict(x)
            
        else:
            
            print fidelity
            raise NameError('Fidlity must be either high or low')
    

    def initialize_pop(self, n_pop):
        
        '''
        Create a random initial population of size n_pop
        '''
        
        self.pop = []
        
        for i in xrange(n_pop):
            
            self.pop.append( [ -4. + random.random() * 8. for i in xrange(2) ] )
    

    def genetic_algorithm(self, n_gens, frac_controlled = 0.1, n_snaps = 0):

        '''
        Optimize the population using a genetic algorithm
        '''
        
        # Find the frequency at which generations are controlled
        if frac_controlled == 0.0:
            controlled_every = n_gens + 1
        else:
            controlled_every = int( 1. / frac_controlled )
        
        # Initialize list of steps for taking snapshots
        # The population will be shown after the mutation in the listed generations
        # "After generation 0" gives the initial population
        snap_ind = 0
        if n_snaps == 0:
            snap_record = []
        elif n_snaps == 1:
            snap_record = [0]
        else:
            snap_record = [int( float(i) / (n_snaps - 1) * ( n_gens ) ) for i in range(n_snaps)]

        if n_snaps > 0:
            snap_ind += 1
            self.plot_pop(fid = 'low', fname = 'pop_pic_1.png', gen_num = 0)
    
        for i in xrange(n_gens):

            # Evolve population
            if i > 0 and i % controlled_every == 0:
                self.evolve(controlled = True)
            else:
                self.evolve(controlled = False)
            
            # Show the population if it is a snapshot generation
            if i+1 in snap_record:
                snap_ind += 1
                self.plot_pop(fid = 'low', fname = 'pop_pic_' + str(snap_ind) + '.png' , gen_num = i+1)          
            

    def evolve(self, retain=0.2, random_select=0.1, mutate=0.1, controlled = False):
    
        '''
        Execute one generation of the genetic algorithm
        Evolve the population using metation and crossover
        retain: top fraction to keep
        random_select: 
        mutate: 
        controlled: If True evaluate the full model and refine the neural network
                    If false use the neural network
        '''    
        
        # Evaluate fitness values for the population
        if controlled: 
            graded_pop = [ [ self.eval_OF( self.pop[i], fidelity = 'high' ), i ] for i in range( len( self.pop )) ]
        else:
            graded_pop = [ [ self.eval_OF( self.pop[i], fidelity = 'low' ), i ] for i in range( len( self.pop )) ]
            
        graded_pop = sorted(graded_pop)
        sorted_pop = [ self.pop[ graded_pop[i][1] ] for i in range( len( graded_pop ))]    
        self.fitnesses = np.array( [ graded_pop[i][0] for i in range( len( graded_pop ))]     )
        self.fitnesses = self.fitnesses.reshape( -1,1)
        
        # Add data to refine the neural network
        if controlled:
            BF.low_fid.refine(np.array( sorted_pop ), self.fitnesses)
        
        # Should record the best and average fitness values
        
        retain_length = int(len(self.pop) * retain)
        new_pop = sorted_pop[:retain_length]            # Take the bottom scores
        
        # randomly add other individuals to promote genetic diversity
        for individual in sorted_pop[retain_length:]:
            if random_select > random.random():
                new_pop.append(individual)
        
        # Mutate some individuals
        for individual in new_pop:
            if mutate > random.random():
#                print individual
                individual[0] = individual[0] + random.gauss(0, 0.5)    # Add Gaussian noise
                individual[1] = individual[1] + random.gauss(0, 0.5)    # Add Gaussian noise
          
        # Crossover parents to create children    
        desired_length = len(self.pop) - len(new_pop)
        children = []
        while len(children) < desired_length:
            
            momdad = random.sample(new_pop, 2)        
            Dad = momdad[0]
            Mom = momdad[1]
            children.append( [ Dad[0], Mom[1] ] )
    
        new_pop.extend(children)
        
        self.pop = new_pop
    
    
    def plot_pop(self, fname = None, fid = 'high', gen_num = None):
        
        '''
        Plot a heat map of the objective function
        '''     
        
        plt.figure() 
        # Make data.
        x1min = -4
        x1max = 4
        x2min = -4
        x2max = 4
        
        X = np.linspace(x1min, x1max, 50)
        Y = np.linspace(x2min, x2max, 50)
        X, Y = np.meshgrid(X, Y)
    
        Z = np.zeros([ X.shape[0], X.shape[1] ])
        for i in range(len(X)):
            for j in range(len(Y)):
                Z[i,j] = self.eval_OF( [ X[i,j] , Y[i,j] ], fidelity = fid )
                
#        plt.contourf(X, Y, Z, 15, cmap=plt.cm.rainbow, vmax=Z.max(), vmin=Z.min())
        plt.contourf(X, Y, Z, 15, cmap=plt.cm.rainbow, vmax=13, vmin=0)
        plt.colorbar()
        
        
        '''
        Plot the values of each individual in the population
        '''    
        
        if not self.pop is None:
        
            x1_vec = [ind[0] for ind in self.pop]
            x2_vec = [ind[1] for ind in self.pop]
            
            plt.plot(x1_vec, x2_vec, marker='o', color = 'k', linestyle = 'None')       # population
        #    plt.plot(-1, 2, marker='x', color = 'r', linestyle = 'None')                # target
        
        plt.xlabel('$x_1$',size=24)
        plt.ylabel('$x_2$',size=24)
        plt.xticks(size=20)
        plt.yticks(size=20)
        plt.xlim([-4, 4])
        plt.ylim([-4, 4])
        if not gen_num is None:
            plt.title('Generation ' + str(gen_num))
        plt.tight_layout()
        
        if fname is None:
            plt.show()
        else:
            plt.savefig(fname)
            plt.close()
    

if __name__ == "__main__":
    
    os.system('clear')
    
    BF = Bifidelity(pop_size = 10) 
    BF.low_fid = NeuralNetwork()
    
    '''
    Generate initial training data
    '''
    
    x1min = -4
    x1max = 4
    x2min = -4
    x2max = 4
    
    n_X1 = 4
    n_X2 = 4
    
    X1 = np.linspace(x1min, x1max, n_X1)
    X2 = np.linspace(x2min, x2max, n_X2)
    X1, X2 = np.meshgrid(X1, X2)
    X1 = X1.reshape(-1,1)
    X2 = X2.reshape(-1,1)
#    X1 = X1.reshape([ n_X1 * n_X2 ])
#    X2 = X2.reshape([ n_X1 * n_X2 ])
    X = np.hstack( [X1, X2] )
    
    Y = np.zeros([ X.shape[0], 1 ])
    for i in range( X.shape[0] ):
        Y[i] = BF.eval_OF( X[i,:] )
        
    # Train neural network and plot high and low fidelity objective functions
    BF.low_fid = NeuralNetwork()
    BF.low_fid.refine(X, Y)
    BF.plot_pop(fname = 'high_fid.png', fid = 'high')
    BF.plot_pop(fname = 'low_fid.png', fid = 'low')

    
    '''
    Create model and execute genetic algorithm
    '''

    # Numerical parameters
    p_count = 100                   # population size    
    n_gens = 100                    # number of generations
    fc = 0.1           # fraction of generations that are controlled

    
    BF.genetic_algorithm(n_gens, frac_controlled = 0.1, n_snaps = 10)