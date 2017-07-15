# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 14:03:35 2017

@author: mpnun
"""

import numpy as np
import random
import time
import matplotlib.pyplot as plt


#class GA_individual(object):
#    
#    def __init__(self):
#        pass
#    
#    def 

class MOGA():
    
    '''
    Multi-objective genetic algorithm
    Handles two objective functions
    '''
    
    def __init__(self):
        
        '''
        Initialize with an empty population
        '''
        
        self.P = None                       # Population: List of individuals
        self.Q = None                       # candidate population
        self.fitnesses = None               # Fitness values for the individuals in the population
        


    def randomize_pop(self):
        
        '''
        Create a random initial population of size n_pop
        '''
        
        if self.P is None:
            raise NameError('Population has not been initialized.')
            
        for indiv in self.P:
            indiv.randomize()
    

#    def eval_pop(self):
#        
#        '''
#        Evaluate the objective functions for all individuals in the population
#        '''
#        
#        graded_pop = [ [ 0, 0, i ] for i in range( len( self.P )) ]
#        for i in range( len( self.P )):
#            score = self.P[i].eval_OFs()
#            graded_pop[i][0] = score[0]
#            graded_pop[i][1] = score[1]
            

    def genetic_algorithm(self, n_gens, n_snaps = 0):

        '''
        Optimize the population using a genetic algorithm
        '''
        
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

#        if n_snaps > 0:
#            snap_ind += 1
#            self.plot_pop(fname = 'pop_pic_1.png', gen_num = 0)
    
        CPU_start = time.time()
        for i in xrange(n_gens):

            # Evolve population
            self.evolve()
            
            # Show the population if it is a snapshot generation
            if i+1 in snap_record:
                snap_ind += 1
                self.plot_pop(fname = 'pop_pic_' + str(snap_ind) + '.png' , gen_num = i+1)
                
        CPU_end = time.time()
        print('Time elapsed: ' + str(CPU_end - CPU_start) )
            

    def create_candidates(self):
        
        '''
        Use the population P to create the candidate population Q
        '''
        self.Q = []

    def evolve(self, retain=0.2, random_select=0.1, mutate=0.3, sigma = 1., controlled = False):
    
        '''
        Execute one generation of the genetic algorithm
        Evolve the population using metation and crossover
        retain: top fraction to keep
        random_select: 
        mutate: 
        controlled: If True evaluate the full model and refine the neural network
                    If false use the neural network
        Uses the algorithm from K. Deb, S. Pratab, S. Agarwal, and T. Meyarivan, IEEE Trans. Evol. Comput. 6, 182 (2002).
        '''    
#        print self.P

#        R = self.P + self.Q

        # Evaluate fitness values for the population
        graded_pop = [ [ 0, 0, i ] for i in range( len( self.P )) ]
        for i in range( len( self.P )):
            score = self.P[i].eval_OFs()
            graded_pop[i][0] = score[0]
            graded_pop[i][1] = score[1]
            
        '''
        Find domination relationships
        '''
        n = [0 for i in range( len( self.P ) ) ]                      # Number of individuals which dominate n
        S = [ [] for i in range( len( self.P ) ) ]                    # Set of individuals which the individual dominates
        for i in range( len( self.P )):           # i in [0,1,...,n-1]
            for j in range(i):                      # j < i
                
                if graded_pop[i][0] < graded_pop[j][0] and graded_pop[i][1] < graded_pop[j][1]:             # i dominates j
                    n[j] += 1
                    S[i].append(j)
                elif graded_pop[j][0] < graded_pop[i][0] and graded_pop[j][1] < graded_pop[i][1]:           # j dominates i
                    n[i] += 1
                    S[j].append(i)
                else:                                                                                       # neither dominates the other
                    pass
        
        '''
        Identify the various levels of Pareto fronts
        '''
        Fronts = [ [] ]
        ranks = [0 for i in range( len( self.P ) ) ]
        for i in range( len( self.P )):
            if n[i] == 0:
                Fronts[0].append(i)
                ranks[i] = 0
        
        f_count = 0
        next_front = True
        while next_front:
            
            # Fill in the next front
            Q = []
            for i in Fronts[f_count]:
                for j in S[i]:
                    n[j] -= 1
                    if n[j] == 0:
                        Q.append(j)
                        ranks[j] = f_count+1
                        
            f_count += 1
            
            if Q == []:
                next_front = False
            else:
                Fronts.append(Q)
        
        
        '''
        Assign distance metric for each individual in each front
        '''
        
        graded_pop_arr = np.array( graded_pop )        
        dist_met = [0 for i in range( len( self.P ) ) ]       # Average distance from adjacent individuals on its front

        for f in Fronts:                                        # Loop through each front
            
            sub_graded_arr = graded_pop_arr[f, :]
            
            for m in [0,1]:                                     # m: index of the objective we are considering
            
                sorted_data = sub_graded_arr[sub_graded_arr[:, m].argsort()]        # sort according to data in the m'th objective
                dist_met[ int( sorted_data[ 0, -1] ) ] = float('inf')
                dist_met[ int( sorted_data[-1, -1] ) ] = float('inf')
                
                for ind in range(1, sorted_data.shape[0]-1):
                    if sorted_data[-1, m] - sorted_data[0, m] > 0:      # Accounts for the case of no diversity in one of the objectives
                        dist_met[ int( sorted_data[ind,-1] ) ] += ( sorted_data[ind+1, m] - sorted_data[ind-1, m] ) / ( sorted_data[-1, m] - sorted_data[0, m] )       # Add normalized distance to nearest neighbors                
            
            # Sort the front accoeding to distances
            to_sort = [ [dist_met[f[i]] , f[i] ] for i in range(len(f))]
            to_sort = sorted(to_sort, reverse=True)
            f = [to_sort[i][1] for i in range(len(f))]
            
        
        # Should sort according to distance
        # Should record the best and average fitness values

        retain_length = int(len(self.P) * retain)
        new_pop = []
        front_ind = 0
        ind_in_front = 0
        while len(new_pop) < retain_length:
            new_pop.append( self.P[ Fronts[front_ind][ind_in_front] ] )
            ind_in_front += 1
            if ind_in_front >= len(Fronts[front_ind]):
                front_ind += 1
                ind_in_front = 0
        
        # randomly add other individuals to promote genetic diversity
#        for individual in sorted_pop[retain_length:]:
#            if random_select > random.random():
#                new_pop.append(individual)
        
        # Mutate some individuals
        for individual in new_pop:
            if mutate > random.random():
                individual.mutate()
          
        # Crossover parents to create children    
        desired_length = len(self.P) - len(new_pop)
        children = []
        while len(children) < desired_length:
            
            momdad = random.sample(new_pop, 2)        
            Dad = momdad[0]
            Mom = momdad[1]
            children.append( Dad.crossover(Mom) )
    
        new_pop.extend(children)
        
        self.P = new_pop
    
    
    def plot_pop(self, fname = None, gen_num = None):
        
        '''
        Plot the objective function values of each individual in the population
        '''    

        grades = [[0, 0] for i in range(len(self.P))]
        for i in range(len(self.P)):
            OFs = self.P[i].eval_OFs()
            grades[i] = OFs
            
        self.fitnesses = np.array(grades)

        plt.plot(self.fitnesses[:,0].reshape(-1,1), self.fitnesses[:,1].reshape(-1,1), marker='o', color = 'k', linestyle = 'None')       # population
        plt.xlabel('$y_1$',size=24)
        plt.ylabel('$y_2$',size=24)
        plt.xticks(size=20)
        plt.yticks(size=20)
#        plt.xlim([-4, 4])
#        plt.ylim([-4, 4])
        if not gen_num is None:
            plt.title('Generation ' + str(gen_num))
        plt.tight_layout()
        
        if fname is None:
            plt.show()
        else:
            plt.savefig(fname)
            plt.close()
            
        # Should also show the x values for the best individual so far