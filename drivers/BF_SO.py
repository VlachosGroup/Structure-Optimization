import sys
sys.path.append('/home/vlachos/mpnunez/Github/Structure-Optimization/ORR/')
sys.path.append('/home/vlachos/mpnunez/Github/Zacros-Wrapper')
sys.path.append('/home/vlachos/mpnunez/python_packages/ase')
sys.path.append('/home/vlachos/mpnunez/python_packages/sklearn/lib/python2.7/site-packages')
sys.path.append('/home/vlachos/mpnunez/python_packages/DEAP/lib/python2.7/site-packages')

import numpy
import multiprocessing
import random
import time
import numpy as np

from deap import creator, base, tools, algorithms

#from Cat_structure import cat_structure
from cat_optimize import cat_optimize
from NeuralNetwork import NeuralNetwork
from plotpop import *

''' User input '''
d1 = 12
d2 = 12
pop_size = 200
offspring_size = 1000
n_gens = 300
n_snaps = 10
crossover_prob = 0.3
mutation_prob = 0.2
mutation_rate = 1.5/(d1*d2)

#pop_size = 20
#offspring_size = 100
#n_gens = 30

'''
Create individual
'''
creator.create('FitnessMulti', base.Fitness, weights = [1.0,])
creator.create('Individual', list, fitness=creator.FitnessMulti)

'''
Evaluation methods
'''
eval_obj = cat_optimize()
def evalFitness_HF(individual):
    return eval_obj.eval_x( individual )[1],

surrogate = NeuralNetwork()
def evalFitness_LF(individual):
    return surrogate.predict( np.array( [individual] ) )[0],

def cross(ind1,ind2):
    return eval_obj.geo_crossover( ind1,ind2 )
    
'''
Populate toolbox with operators
'''
toolbox = base.Toolbox()
toolbox.register('bit', random.randint, 0, 1)
toolbox.register('individual', tools.initRepeat, creator.Individual, toolbox.bit, n=d1*d2)
toolbox.register('population', tools.initRepeat, list, toolbox.individual, n=pop_size)
toolbox.register('evaluate_HF', evalFitness_HF)
toolbox.register('evaluate_LF', evalFitness_LF)
#toolbox.register('mate', tools.cxTwoPoint)
toolbox.register('mate', cross)
toolbox.register('mutate', tools.mutFlipBit, indpb=mutation_rate )
#toolbox.register('select', tools.selBest)
#toolbox.register('select', tools.selRoulette)
toolbox.register('select', tools.selTournament, tournsize=2)

hof = tools.HallOfFame(1)               # Record the best individual as the optimization progresses

'''
Register statistics
'''
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("min", numpy.min)
stats.register("avg", numpy.mean)
stats.register("max", numpy.max)

'''
Enable parallelization
'''
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

x_data = []
y_data = []

bifidelity = False
CPU_start = time.time()


for gen in range(n_gens):
    
    controlled = (gen % n_snaps == 0) and bifidelity
    
    # Evaluate population fitnesses if it is the first iteration or if it is a bifidelity controlled generation
    if controlled or gen == 0:
        
        #CPU_start_eval = time.time()
        fits = toolbox.map(toolbox.evaluate_HF, population)
        for fit, ind in zip(fits, population):
            ind.fitness.values = fit
        #CPU_end_eval = time.time()
        #print('Evaluation time: ' + str(CPU_end_eval - CPU_start_eval) + ' seconds')
    
    # Refine neural network model
    if gen % n_snaps == 0:
        fits = toolbox.map(toolbox.evaluate_HF, population)
        for fit, ind in zip(fits, population):
            ind.fitness.values = fit
            
        for ind in population:
            x_data.append(ind)
        y_data = y_data + fits
        
    if controlled:
        surrogate.refine( np.array(population), np.array(fits) )
        surrogate.plot_parity('parity_' + str(gen) + '.png', title = 'Generation ' + str(gen))
    
    if gen % (n_gens / n_snaps) == 0:
        print str(gen) + ' generations have elapsed - taking snapshot'
        record = stats.compile(population)
        print record
        plot_pop_SO(np.array(fits), fname = 'Generation_' + str(gen) + '.png', title = 'Generation ' + str(gen))
    
    '''
    Create offspring form the population
    '''
    #offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.1)
    #offspring = algorithms.varOr(population, toolbox, lambda_=offspring_size, cxpb=crossover_prob, mutpb=mutation_prob)
    
    offspring = []
    while len(offspring) < offspring_size:
        op_choice = random.random()
        if op_choice < crossover_prob:            # Apply crossover
            ind1, ind2 = map(toolbox.clone, random.sample(population, 2))
            ind1, ind2 = toolbox.mate(ind1, ind2)
            del ind1.fitness.values
            offspring.append(ind1)
            if len(offspring) < offspring_size:
                del ind2.fitness.values
                offspring.append(ind2)
        else:                           # Apply reproduction
            offspring.append(random.choice(population))
    
    for ind in offspring:
        op_choice = random.random()
        if op_choice < mutation_prob:  # Apply mutation
            ind, = toolbox.mutate(ind)
            del ind.fitness.values
    
    # Evaluate offspring with low fidelity model
    if bifidelity:
        for ind in offspring:
            ind.fitness.values = evalFitness_LF(ind)
    
    # Evaluate offspring with high fidelity model
    else:
        fits = toolbox.map(toolbox.evaluate_HF, offspring)
        for fit, ind in zip(fits, offspring):
            ind.fitness.values = fit
    
    #population = toolbox.select(offspring + population, k = pop_size)           # (mu + lambda)-selection
    population = toolbox.select(offspring, k = pop_size)                 # (mu,lambda)-selection
    hof.update(population)

'''
Evaluate final population
'''
fits = toolbox.map(toolbox.evaluate_HF, population)
for fit, ind in zip(fits, population):
    ind.fitness.values = fit
gen = gen + 1
for ind in population:
    x_data.append(ind)
y_data = y_data + fits
#surrogate.refine( np.array(population), np.array(fits) )
#surrogate.plot_parity('parity_' + str(gen) + '.png', title = 'Generation ' + str(gen))
print str(gen) + ' generations have elapsed - taking snapshot'
record = stats.compile(population)
print record
plot_pop_SO(np.array(fits), fname = 'Generation_' + str(gen) + '.png', title = 'Generation ' + str(gen))    

CPU_end = time.time()
print('Genetic algorithm time: ' + str(CPU_end - CPU_start) + ' seconds')

#np.save('X.npy', np.array(x_data))
#np.save('Y.npy', np.array(y_data))

print hof
     
# Print best activity
#eval_obj.show(x = population[most_active_ind], n_struc = 1)