import sys
sys.path.append('/home/vlachos/mpnunez/Github/Structure-Optimization/ORR/')
sys.path.append('/home/vlachos/mpnunez/Github/Zacros-Wrapper')
sys.path.append('/home/vlachos/mpnunez/python_packages/ase')
sys.path.append('/home/vlachos/mpnunez/python_packages/sklearn/lib/python2.7/site-packages')
sys.path.append('/home/vlachos/mpnunez/python_packages/DEAP/lib/python2.7/site-packages')

from Cat_structure import cat_structure

import multiprocessing
import random
import time
from deap import creator, base, tools, algorithms
import numpy as np
from NeuralNetwork import NeuralNetwork
from plotpop import *


''' User input '''
d1 = 12
d2 = 12

'''
Create individual
'''
creator.create('FitnessMulti', base.Fitness, weights = [1.0,])
creator.create('Individual', list, fitness=creator.FitnessMulti)

'''
Evaluation methods
'''
eval_obj = cat_structure(met_name = 'Pt', facet = '111', dim1 = d1, dim2 = d2)
def evalFitness_HF(individual):
    return eval_obj.eval_x( individual )[1],

surrogate = NeuralNetwork()
def evalFitness_LF(individual):
    return surrogate.predict( np.array( [individual] ) )[0],
    
'''
Populate toolbox with operators
'''
toolbox = base.Toolbox()
toolbox.register('bit', random.randint, 0, 1)
toolbox.register('individual', tools.initRepeat, creator.Individual, toolbox.bit, n=d1*d2)
toolbox.register('population', tools.initRepeat, list, toolbox.individual, n=600)
toolbox.register('evaluate_HF', evalFitness_HF)
toolbox.register('evaluate_LF', evalFitness_LF)
toolbox.register('mate', tools.cxTwoPoint)
toolbox.register('mutate', tools.mutFlipBit, indpb=2./(d1*d2) )
toolbox.register('select', tools.selTournament, tournsize=2)

# Enable parallelization
pool = multiprocessing.Pool()
toolbox.register("map", pool.map)
print 'Using ' + str(multiprocessing.cpu_count()) + ' processors.'

# Initialize population and evaluate
population = toolbox.population()

# Randomize according to coverages
for i in xrange(len(population)):
    coverage = float(i) / len(population[i])
    for j in range(len(population[i])):
        if random.random() < coverage:
            population[i][j] = 1
        else:
            population[i][j] = 0

'''
Main optimization loop
'''

CPU_start = time.time()
for gen in range(100):

    print str(gen) + ' generations have elapsed.'
    controlled = gen % 10 == 0
    
    if controlled:
        
        CPU_start_eval = time.time()
        fits = toolbox.map(toolbox.evaluate_HF, population)
        for fit, ind in zip(fits, population):
            ind.fitness.values = fit
        CPU_end_eval = time.time()
        print('Evaluation time: ' + str(CPU_end_eval - CPU_start_eval) + ' seconds')
        
        surrogate.refine( np.array(population), np.array(fits) )
        surrogate.plot_parity('parity_' + str(gen) + '.png', title = 'Generation ' + str(gen))
        plot_pop_SO(np.array(fits), fname = 'Generation_' + str(gen) + '.png', title = 'Generation ' + str(gen))
    
    offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.1)
    
    bifidelity = True
    if bifidelity:
        for ind in offspring:
            ind.fitness.values = evalFitness_LF(ind)
    else:
        fits = toolbox.map(toolbox.evaluate_HF, offspring)
        for fit, ind in zip(fits, offspring):
            ind.fitness.values = fit
    
    population = toolbox.select(offspring + population, k = 300)

'''
Evaluate final population
'''
fits = toolbox.map(toolbox.evaluate_HF, population)
for fit, ind in zip(fits, population):
    ind.fitness.values = fit
gen = 100
surrogate.refine( np.array(population), np.array(fits) )
surrogate.plot_parity('parity_' + str(gen) + '.png', title = 'Generation ' + str(gen))  
plot_pop_SO(np.array(fits), fname = 'Generation_' + str(gen) + '.png', title = 'Generation ' + str(gen))    

CPU_end = time.time()
print('Genetic algorithm time: ' + str(CPU_end - CPU_start) + ' seconds')

most_active_ind = 0
best_activity = fits[0]
for ind in range(len(population)):
    if fits[ind] > best_activity:
        most_active_ind = ind
        best_activity = fits[ind]
     
# Print best activity        
print 'Highest activity: ' + str( np.max(fits) ) + ' mA/cm^2'
eval_obj.show(x = population[most_active_ind], n_struc = 1)