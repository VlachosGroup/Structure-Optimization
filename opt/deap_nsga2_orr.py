import sys
sys.path.append('/home/vlachos/mpnunez/Github/Structure-Optimization/ORR/')
from Cat_structure import cat_structure

import multiprocessing
import random
from deap import creator, base, tools, algorithms

'''
Create individual
'''
creator.create('FitnessMulti', base.Fitness, weights = [-1.0, 1.0])
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
toolbox.register('population', tools.initRepeat, list, toolbox.individual, n=300)
toolbox.register('evaluate', evalFitness)
toolbox.register('mate', tools.cxTwoPoint)
toolbox.register('mutate', tools.mutFlipBit, indpb=2./144)
toolbox.register('select', tools.selNSGA2)

# Enable parallelization
pool = multiprocessing.Pool()
toolbox.register("map", pool.map)
print 'Using ' + str(multiprocessing.cpu_count()) + ' processors.'

# Evaluate initial population
population = toolbox.population()
fits = toolbox.map(toolbox.evaluate, population)
for fit, ind in zip(fits, population):
    ind.fitness.values = fit
    
    
'''
Main optimization loop
'''
for gen in range(1000):
    print str(gen)
    offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.1)
    fits = toolbox.map(toolbox.evaluate, offspring)
    for fit, ind in zip(fits, offspring):
        ind.fitness.values = fit
    population = toolbox.select(offspring + population, k = 300)

for f in fits:    
    print f