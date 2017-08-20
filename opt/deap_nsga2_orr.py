from Cat_structure import cat_structure

import random
from deap import creator, base, tools, algorithms
creator.create('FitnessMulti', base.Fitness, weights = [-1.0, 1.0])
creator.create('Individual', list, fitness=creator.FitnessMulti)

eval_obj = cat_structure(met_name = 'Pt', facet = '111', dim1 = 12, dim2 = 12)
def evalFitness(individual):
    return eval_obj.eval_x( individual )
    
toolbox = base.Toolbox()
toolbox.register('bit', random.randint, 0, 1)
toolbox.register('individual', tools.initRepeat, creator.Individual, toolbox.bit, n=144)
toolbox.register('population', tools.initRepeat, list, toolbox.individual, n=100)
toolbox.register('evaluate', evalFitness)
toolbox.register('mate', tools.cxUniform, indpb=0.1)
toolbox.register('mutate', tools.mutFlipBit, indpb=0.05)
toolbox.register('select', tools.selNSGA2)

population = toolbox.population()
fits = toolbox.map(toolbox.evaluate, population)
for fit, ind in zip(fits, population):
    ind.fitness.values = fit
    
for gen in range(50):
    offspring = algorithms.varOr(population, toolbox, lambda_=100, cxpb=0.5, mutpb=0.1)
    fits = toolbox.map(toolbox.evaluate, offspring)
    for fit, ind in zip(fits, offspring):
        ind.fitness.values = fit
    population = toolbox.select(offspring + population, k = 100)
    print str(gen)

for f in fits:    
    print f