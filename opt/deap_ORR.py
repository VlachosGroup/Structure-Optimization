import array
import random
import numpy
import multiprocessing

from deap import algorithms
from deap import base
from deap import creator
from deap import tools


import sys
sys.path.append('/home/vlachos/mpnunez/Github/Structure-Optimization/ORR/')
from Cat_structure import cat_structure

''' User input section '''
dim = 12
ngens = 1000
popsize = 208
cross_prob = 0.5
mut_prob = 0.2
''' '''

if __name__ == "__main__":

    '''
    Individual
    '''
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", array.array, typecode='b', fitness=creator.FitnessMax)
    
    '''
    Evaluation
    '''
    eval_obj = cat_structure(met_name = 'Pt', facet = '111', dim1 = dim, dim2 = dim)
    def evalOneMax(individual):
        OFs = eval_obj.eval_x( individual )
        return OFs[1],
    
    '''
    Toolbox
    '''
    toolbox = base.Toolbox()
    
    # Attribute generator
    toolbox.register("attr_bool", random.randint, 0, 1)
    
    # Structure initializers
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, dim**2)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evalOneMax)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=2./dim**2)
    toolbox.register("select", tools.selTournament, tournsize=2)


    # Enable parallelization
    pool = multiprocessing.Pool()
    toolbox.register("map", pool.map)
    print 'Using ' + str(multiprocessing.cpu_count()) + ' processors.'
    
    '''
    Main optimization
    '''
    
    random.seed(64)
    
    pop = toolbox.population(n=popsize)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)
    
    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=cross_prob, mutpb=mut_prob, ngen=ngens, 
                                   stats=stats, halloffame=hof, verbose=True)
    
    #return pop, log, hof