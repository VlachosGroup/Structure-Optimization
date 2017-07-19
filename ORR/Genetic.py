# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 14:03:35 2017

@author: mpnun
"""

import numpy as np
import random
import time
import matplotlib.pyplot as plt


class MOGA_individual(object):
    
    '''
    Template class for an individual used in the genetic algorithm
    '''
    
    def __init__(self):
        pass

    def randomize(self):
        pass
         
    def get_OFs(self):
        pass
    
    def eval_OFs(self):
        pass
     
    def mutate(self):
        pass
         
    def crossover(self, mate):
        pass


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


    def randomize_pop(self):
        
        '''
        Create a random initial population of size n_pop
        '''
        
        if self.P is None:
            raise NameError('Population has not been initialized.')
            
        for indiv in self.P: 
            indiv.randomize()
            

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

        if n_snaps > 0:
            snap_ind += 1
            self.plot_pop(fname = 'pop_pic_1.png', gen_num = 0)
    
        CPU_start = time.time()
        for i in xrange(n_gens):
            print 'Generation ' + str(i)
            # Evolve population
            self.evolve()
            
            # Show the population if it is a snapshot generation
            if i+1 in snap_record:
                snap_ind += 1
                self.plot_pop(fname = 'pop_pic_' + str(snap_ind) + '.png' , gen_num = i+1)
                
        CPU_end = time.time()
        print('Time elapsed: ' + str(CPU_end - CPU_start) + ' seconds')
            

    def evolve(self):
    
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

        N = len(self.P)
        R = self.P
        if not self.Q is None:          # self.Q is None for the 1st generation
            R = self.P + self.Q

        # Given R, we need to dermine the new P

        # Extract fitness values for R
        graded_pop = [ [ 0, 0, i ] for i in range( len( R )) ]
        for i in range( len( R )):
            score = R[i].get_OFs()
            graded_pop[i][0] = score[0]
            graded_pop[i][1] = score[1]
            
        '''
        Find domination relationships
        '''
        n = [0 for i in range( len( R ) ) ]                      # Number of individuals which dominate n
        S = [ [] for i in range( len( R ) ) ]                    # Set of individuals which the individual dominates
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
        
        # Identify the first front
        Fronts = [ [] ]
        ranks = [0 for i in range( len( R ) ) ]
        for i in range( len( R )):
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
        dist_met = [0 for i in range( len( R ) ) ]       # Average distance from adjacent individuals on its front

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
            
        
        '''
        Build the new P
        '''

        self.P = []
        P_indices = []
        front_ind = 0
        ind_in_front = 0
        while len(self.P) < N:
            self.P.append( R[ Fronts[front_ind][ind_in_front] ] )
            P_indices.append( Fronts[front_ind][ind_in_front] )
            ind_in_front += 1
            if ind_in_front >= len(Fronts[front_ind]):
                front_ind += 1
                ind_in_front = 0
        
        '''
        Given the new P, use tournament selection, mutation and crossover to create Q
        '''
        
        self.Q = []
        frac_mutate = 0.5       # fraction of Q filled with mutations. The rest is filled with crossover.
        
        # Tournament select parents and mutate to create children
        while len(self.Q) < int(N * frac_mutate):
            
            contestants = random.sample(P_indices, 2)   # Need to implment tournament selection here instead      
            if ranks[contestants[0]] < ranks[contestants[1]]:
                chosen_one = R[contestants[0]]
            elif ranks[contestants[1]] < ranks[contestants[0]]:
                chosen_one = R[contestants[1]]
            else:
                if dist_met[contestants[0]] <= dist_met[contestants[1]]:
                    chosen_one = R[contestants[0]]
                else:
                    chosen_one = R[contestants[1]]

            self.Q.append( chosen_one.mutate() )
          
        # Crossover parents to create children
        while len(self.Q) < N:
            
            # Tournament select to choose Mom
            contestants = random.sample(P_indices, 2)   # Need to implment tournament selection here instead      
            if ranks[contestants[0]] < ranks[contestants[1]]:
                Mom = R[contestants[0]]
            elif ranks[contestants[1]] < ranks[contestants[0]]:
                Mom = R[contestants[1]]
            else:
                if dist_met[contestants[0]] <= dist_met[contestants[1]]:
                    Mom = R[contestants[0]]
                else:
                    Mom = R[contestants[1]]
            
            # Tournament select to choose Dad
            contestants = random.sample(P_indices, 2)   # Need to implment tournament selection here instead      
            if ranks[contestants[0]] < ranks[contestants[1]]:
                Dad = R[contestants[0]]
            elif ranks[contestants[1]] < ranks[contestants[0]]:
                Dad = R[contestants[1]]
            else:
                if dist_met[contestants[0]] <= dist_met[contestants[1]]:
                    Dad = R[contestants[0]]
                else:
                    Dad = R[contestants[1]]
            
            # Crossover Mom and Dad to create a child
            self.Q.append( Dad.crossover(Mom) )
            if len(self.Q) < N:                     # Crossover the other way of there is still room
                self.Q.append( Mom.crossover(Dad) )
    
    
    def plot_pop(self, fname = None, gen_num = None):
        
        '''
        Plot the objective function values of each individual in the population
        '''    

        grades = [[0, 0] for i in range(len(self.P))]
        for i in range(len(self.P)):
            grades[i] = self.P[i].get_OFs()
            
        fitnesses = np.array(grades)

        plt.plot(fitnesses[:,0].reshape(-1,1), fitnesses[:,1].reshape(-1,1), marker='o', color = 'k', linestyle = 'None')       # population
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