# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 14:03:35 2017

@author: mpnun
"""

import numpy as np
import random

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

from genetic import *

def Ackley(x):
    
    '''
    x is an n-dimensional vector
    '''
    
    x = np.array(x)
    return -20 * np.exp( -0.2 * np.sqrt( np.mean( x ** 2 ) ) ) - np.exp( np.mean( np.cos( 2 * np.pi * x ) ) ) + 20 + np.e


def simple_OF(x):
    
    return np.sqrt( ( x[0] + 1 ) ** 2 + (x[1] - 2 ) ** 2 )

    
def plot_Ackley():
    
    '''
    Plot the objective function. It would be better to do this in 2-D with colors...
    '''    
    
    # Make data.
    x1min = -4
    x1max = 4
    x2min = -4
    x2max = 4
    
    X = np.arange(x1min, x1max, 0.25)
    Y = np.arange(x2min, x2max, 0.25)
    X, Y = np.meshgrid(X, Y)

    Z = np.zeros([ X.shape[0], X.shape[1] ])
    for i in range(len(X)):
        for j in range(len(Y)):
#            Z[i,j] = Ackley( np.array([ X[i,j] , Y[i,j] ]) )
            Z[i,j] = simple_OF( np.array([ X[i,j] , Y[i,j] ]) )
    
    # Plot the surface.
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    plt.xlabel(r'$x_1$', size = 24)
    plt.ylabel(r'$x_2$', size = 24)
    plt.xticks(size=24)
    plt.yticks(size=24)
#    plt.zlabel('Value')    
    
    # Customize the z axis.
#    ax.set_zlim(-1.01, 1.01)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    
    plt.show()

def rand_indiv():
    
    '''
    Create a member of the population.
    '''
    
    return [ -4. + random.random() * 8. for i in xrange(2) ] 
 

def evolve(pop, retain=0.2, random_select=0.1, mutate=0.1):
    
    '''
    Evolve the population using metation and crossover
    retain: top fraction to keep
    random_select: 
    mutate: 
    '''    
    
    graded = [ [ simple_OF(x), x[0], x[1] ] for x in pop]
    graded = sorted(graded)
    sorted_pop = [ [ x[1], x[2] ] for x in graded]    
    scores = np.array( [ x[0] for x in graded] )
    print '\nBest score: ' + str(scores[0])
    print 'Average score: ' + str(np.mean(scores))
    
    retain_length = int(len(pop) * retain)
    new_pop = sorted_pop[:retain_length]            # Take the bottom scores
    
    # randomly add other individuals to promote genetic diversity
    for individual in sorted_pop[retain_length:]:
        if random_select > random.random():
            new_pop.append(individual)
    
    # Mutate some individuals
    for individual in new_pop:
        if mutate > random.random():
            individual[0] = individual[0] + random.gauss(0, 0.5)    # Add Gaussian noise
            individual[1] = individual[1] + random.gauss(0, 0.5)    # Add Gaussian noise
      
    # Crossover parents to create children    
    desired_length = len(pop) - len(new_pop)
    children = []
    while len(children) < desired_length:
        
        momdad = random.sample(new_pop, 2)        
        Dad = momdad[0]
        Mom = momdad[1]
        children.append( [ Dad[0], Mom[1] ] )

    new_pop.extend(children)
    
    return new_pop
    
    
def plot_pop(p, fname = None):
    
    '''
    Plot the values of each individual in the population
    '''    
    
#    print len(p)
#    for ind in p:
#        print ind
    x1_vec = [ind[0] for ind in p]
    x2_vec = [ind[1] for ind in p]
    plt.figure()
    plt.plot(x1_vec, x2_vec, marker='o', color = 'b', linestyle = 'None')       # population
    plt.plot(-1, 2, marker='x', color = 'r', linestyle = 'None')                # target
    plt.xlabel('$x_1$',size=24)
    plt.ylabel('$x_2$',size=24)
    plt.xticks(size=20)
    plt.yticks(size=20)
    plt.xlim([-4, 4])
    plt.ylim([-4, 4])
    plt.tight_layout()
    
    if fname is None:
        plt.show()
    else:
        plt.savefig(fname)
        plt.close()
    

if __name__ == "__main__":    
    
    # Track the number of evaluations of the Ackley function
    # Plot the objective function as a heat map
    
    # Initialize population
    p_count = 100       # population size
    p = [ rand_indiv() for x in xrange(p_count) ]
    
    plot_pop(p, fname = 'pop_pic_0.png')
   
    for i in xrange(100):
        p = evolve(p)
        
        if i % 10 == 0:
            plot_pop(p, fname = 'pop_pic_' + str(i/10 + 1) + '.png' )