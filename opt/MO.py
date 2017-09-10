import sys
sys.path.append('/home/vlachos/mpnunez/Github/Structure-Optimization/ORR/')
sys.path.append('/home/vlachos/mpnunez/Github/Zacros-Wrapper')
sys.path.append('/home/vlachos/mpnunez/python_packages/ase')
sys.path.append('/home/vlachos/mpnunez/python_packages/sklearn/lib/python2.7/site-packages')
sys.path.append('/home/vlachos/mpnunez/python_packages/DEAP/lib/python2.7/site-packages')

from Cat_structure import cat_structure

import multiprocessing
import random
from deap import creator, base, tools, algorithms
import numpy as np
from plotpop import plot_pop_MO

'''
Create individual
'''
creator.create('FitnessMulti', base.Fitness, weights = [-1.0, 3.0])
creator.create('Individual', list, fitness=creator.FitnessMulti)

'''
Evaluation method
'''
eval_obj = cat_structure(met_name = 'Pt', facet = '111', dim1 = 12, dim2 = 12)
def evalFitness(individual):
    return eval_obj.eval_x( individual )

'''
Populate toolbox with operators
'''    
toolbox = base.Toolbox()
toolbox.register('bit', random.randint, 0, 1)
toolbox.register('individual', tools.initRepeat, creator.Individual, toolbox.bit, n=144)
toolbox.register('population', tools.initRepeat, list, toolbox.individual, n=600)
toolbox.register('evaluate', evalFitness)
toolbox.register('mate', tools.cxTwoPoint)
toolbox.register('mutate', tools.mutFlipBit, indpb=2./144)
toolbox.register('select', tools.selNSGA2)

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


fits = toolbox.map(toolbox.evaluate, population)
for fit, ind in zip(fits, population):
    ind.fitness.values = fit


'''
Main optimization loop
'''

plot_pop_MO(np.array(fits), fname = 'Generation_0.png', title = 'Generation 0')

for gen in range(1000):
    #print 'Starting generation ' + str(gen+1)
    offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.1)
    fits = toolbox.map(toolbox.evaluate, offspring)
    for fit, ind in zip(fits, offspring):
        ind.fitness.values = fit
    population = toolbox.select(offspring + population, k = 300)

    if (gen+1) % 100 == 0:
        plot_pop_MO(np.array(fits), fname = 'Generation_' + str(gen+1) + '.png', title = 'Generation ' + str(gen+1))
        
most_active_ind = 0
best_activity = fits[0][0]
for ind in range(len(population)):
    if fits[ind][0] > best_activity:
        most_active_ind = ind
        best_activity = fits[ind][0]
        
print best_activity
eval_obj.show(x = population[most_active_ind], n_struc = 1, fmat = 'xsd')