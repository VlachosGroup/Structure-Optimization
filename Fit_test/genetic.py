# Genetic algorithm functions
# Modified from https://lethain.com/genetic-algorithms-cool-name-damn-simple/

import numpy as np
import random
from operator import add

def rand_indiv():
    
    '''
    Create a member of the population.
    '''
    
    return np.array( [ -4. + random.random() * 8. for i in xrange(2) ] )
 

def evolve(pop, target, retain=0.2, random_select=0.05, mutate=0.01):
    
    '''
    Evolve the population using metation and crossover
    '''    
    
    graded = [ (fitness(x, target), x) for x in pop]
    graded = [ x[1] for x in sorted(graded)]
    retain_length = int(len(graded)*retain)
    parents = graded[:retain_length]
    
    # randomly add other individuals to promote genetic diversity
    for individual in graded[retain_length:]:
        if random_select > random():
            parents.append(individual)
            
    # Mutate some individuals
    for individual in parents:
        if mutate > random():
            pos_to_mutate = randint(0, len(individual)-1)
            # this mutation is not ideal, because it
            # restricts the range of possible values,
            # but the function is unaware of the min/max
            # values used to create the individuals,
            individual[pos_to_mutate] = randint(min(individual), max(individual))
            
    # Crossover parents to create children    
    parents_length = len(parents)
    desired_length = len(pop) - parents_length
    children = []
    while len(children) < desired_length:
        male = randint(0, parents_length-1)
        female = randint(0, parents_length-1)
        if male != female:
            male = parents[male]
            female = parents[female]
            half = len(male) / 2
            child = male[:half] + female[half:]
            children.append(child)        
    parents.extend(children)
    
    return parents