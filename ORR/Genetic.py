# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 14:03:35 2017

@author: mpnun
"""

import numpy as np
import random
import time
import matplotlib.pyplot as plt
from mpi4py import MPI

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
     
    def mutate(self, sigma):
        pass
         
    def crossover(self, mate):
        pass
    
    def show(self, i):
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
        self.P_y1 = None                    # first objective function values for individuals in P, doubles
        self.P_y2 = None
        
        self.Q = None                       # candidate population
        self.Q_y1 = None
        self.Q_y2 = None
        
        self.eval_obj = None
        
        self.COMM = None                    # object for MPI parallelization
        

    def randomize_pop(self):
        
        '''
        Create a random initial population of size n_pop
        '''
        
        if self.P is None:
            raise NameError('Population has not been initialized.')
            
        for indiv in self.P: 
            indiv.randomize()
        
        
    def mutate(self, x):
    
        '''
        Mutates an individual to yield an offspring
        '''
        
        x_child = np.array([x[i] for i in range(len(x))])        # copy the data
        
        for i in range(len(x_child)):
            if np.random.random() < 1.0 / len(x_child):
                if x_child[i] == 0:
                    x_child[i] = 1
                else:
                    x_child[i] = 0
            
        return x_child
        
        
    def crossover(self, x1, x2):
    
        '''
        Crossover with a mate
        Return the child
        '''
        
        return np.hstack( [x1[:len(x1)/2:], x2[len(x1)/2::] ] )
            

    def genetic_algorithm(self, n_gens, n_snaps = 0):

        '''
        Optimize the population using a genetic algorithm
        '''
        
        self.COMM = MPI.COMM_WORLD
        
        # Set different random seeds for each processor
        random.seed(a=12345 + self.COMM.rank)
        np.random.seed(seed=12345 + self.COMM.rank)
        
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

        # Evaluate individuals in P before the 1st generation
        self.P_y1 = []
        self.P_y2 = []
        for p in self.P:
            OFs = self.eval_obj.eval_x(p)
            self.P_y1.append(OFs[0])
            self.P_y2.append(OFs[1])
            
        if n_snaps > 0 and self.COMM.rank == 0:
            snap_ind += 1
            self.plot_pop(fname = 'pop_pic_1.png', gen_num = 0)
    
        CPU_start = time.time()
        for i in xrange(n_gens):
        
            if self.COMM.rank == 0:          # Report progress
                print 'Generation ' + str(i+1)
                
            # Evolve population
            self.evolve()
            
            # Show the population if it is a snapshot generation
            if i+1 in snap_record and self.COMM.rank == 0:
                snap_ind += 1
                self.plot_pop(fname = 'pop_pic_' + str(snap_ind) + '.png' , gen_num = i+1)
                
        CPU_end = time.time()
        print('Time elapsed: ' + str(CPU_end - CPU_start) + ' seconds')
        
        #i = 1
        #for indiv in x.P:
        #    indiv.show(i)
        #    indiv.show(i, fmat = 'xsd')
        #    i += 1
        

    def evolve(self, frac_mutate = 0.8):   
    
        '''
        Execute one generation of the genetic algorithm
        Evolve the population using metation and crossover
        retain: top fraction to keep
        random_select: 
        mutate: 
        Uses the NSGA-II algorithm from K. Deb, S. Pratab, S. Agarwal, and T. Meyarivan, IEEE Trans. Evol. Comput. 6, 182 (2002).
        '''    

        N = self.P.shape[0]         # number of individuals in the population
        N_c = self.P.shape[1]       # number of bits for each individual
        
        # Combine P and Q into R (Q is an empty list in the first generation)
        if self.Q is None:
            R = self.P
            R_y1 = self.P_y1
            R_y2 = self.P_y2
        else:
            R = np.vstack([self.P, self.Q])
            R_y1 = np.hstack([self.P_y1, self.Q_y1])
            R_y2 = np.hstack([self.P_y2, self.Q_y2])

        # Given R, we need to dermine the new P
   
        '''
        Find domination relationships
        '''
        n = [0 for i in range( R.shape[0] ) ]                      # Number of individuals which dominate n
        S = [ [] for i in range( R.shape[0] ) ]                    # Set of individuals which the individual dominates
        for i in range( len( self.P )):           # i in [0,1,...,n-1]
            for j in range(i):                      # j < i
                
                if R_y1[i] < R_y1[j] and R_y2[i] < R_y2[j]:             # i dominates j
                    n[j] += 1
                    S[i].append(j)
                elif R_y1[j] < R_y1[i] and R_y2[j] < R_y2[i]:           # j dominates i
                    n[i] += 1
                    S[j].append(i)
                else:                                                                                       # neither dominates the other
                    pass
        
        '''
        Identify the various levels of Pareto fronts
        '''
        
        # Identify the first front
        Fronts = [ [] ]
        ranks = [0 for i in range( R.shape[0] ) ]
        for i in range( R.shape[0]):
            if n[i] == 0:
                Fronts[0].append(i)
                ranks[i] = 0
        
        # Identify all other fronts
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
        # Extract fitness values for R
        graded_pop = [ [ 0, 0, i ] for i in range( R.shape[0] ) ]
        for i in range( R.shape[0] ):
            graded_pop[i][0] = R_y1[i]
            graded_pop[i][1] = R_y2[i]
            
        graded_pop_arr = np.array( graded_pop )        
        dist_met = [0 for i in range( R.shape[0] ) ]       # Average distance from adjacent individuals on its front

        front_ind = 0
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
            Fronts[front_ind] = [to_sort[i][1] for i in range(len(f))]
            front_ind += 1
            
        
        '''
        Selection: Select individuals for the new P
        '''
        
        self.P = []
        self.P_y1 = []
        self.P_y2 = []
            
        P_indices = []
        available = range( R.shape[0] )
        front_ind = 0
        ind_in_front = 0
        while len(self.P) < N:
        
            self.P.append( R[ Fronts[front_ind][ind_in_front], : ] )
            self.P_y1.append( R_y1[ Fronts[front_ind][ind_in_front] ] )
            self.P_y2.append( R_y2[ Fronts[front_ind][ind_in_front] ] )
            
            P_indices.append( Fronts[front_ind][ind_in_front] )
            available.remove( Fronts[front_ind][ind_in_front] )
            ind_in_front += 1
            if ind_in_front >= len(Fronts[front_ind]):
                front_ind += 1
                ind_in_front = 0
        
        self.P = np.array(self.P)
        self.P_y1 = np.array(self.P_y1)
        self.P_y2 = np.array(self.P_y2)
        
        '''
        Given the new P, use tournament selection, mutation and crossover to create Q
        '''        
        
        if not N % self.COMM.size == 0:
            raise NameError('Population size must be an integer multiple of the number of processors.')
        
        my_N = N / self.COMM.size                        # self.COMM.size should divide N
        my_Q = np.array([])
        my_Q_y1 = []
        my_Q_y2 = []

        # Mutation: Tournament select parents and mutate to create children
        while len(my_Q) < int(my_N * frac_mutate) * N_c:
            
            contestants = random.sample(P_indices, 2)      
            if ranks[contestants[0]] < ranks[contestants[1]]:
                chosen_one = contestants[0]
            elif ranks[contestants[1]] < ranks[contestants[0]]:
                chosen_one = contestants[1]
            else:
                if dist_met[contestants[0]] <= dist_met[contestants[1]]:
                    chosen_one = contestants[0]
                else:
                    chosen_one = contestants[1]

            new_candidate = self.mutate( R[chosen_one,:] )
            my_Q = np.hstack([my_Q, new_candidate])
            if np.array_equal(new_candidate, R[chosen_one,:]):      # Mutation may not have changed, so we do not reevaluate the objective functions
                my_Q_y1.append(R_y1[chosen_one])
                my_Q_y2.append(R_y2[chosen_one])
            else:
                OFs = self.eval_obj.eval_x(new_candidate)
                my_Q_y1.append(OFs[0])
                my_Q_y2.append(OFs[1])
          
        # Crossover: Crossover parents to create children
        while len(my_Q) < my_N * N_c:
            
            # Tournament select to choose Mom
            contestants = random.sample(P_indices, 2)    
            if ranks[contestants[0]] < ranks[contestants[1]]:
                Mom = R[contestants[0],:]
            elif ranks[contestants[1]] < ranks[contestants[0]]:
                Mom = R[contestants[1],:]
            else:
                if dist_met[contestants[0]] <= dist_met[contestants[1]]:
                    Mom = R[contestants[0],:]
                else:
                    Mom = R[contestants[1],:]
            
            # Tournament select to choose Dad
            contestants = random.sample(P_indices, 2)
            if ranks[contestants[0]] < ranks[contestants[1]]:
                Dad = R[contestants[0],:]
            elif ranks[contestants[1]] < ranks[contestants[0]]:
                Dad = R[contestants[1],:]
            else:
                if dist_met[contestants[0]] <= dist_met[contestants[1]]:
                    Dad = R[contestants[0],:]
                else:
                    Dad = R[contestants[1],:]
            
            # Crossover Mom and Dad to create a child
            child1 = self.crossover(Dad, Mom)
            child1 = self.mutate(child1)
            my_Q = np.hstack([my_Q, child1])
            OFs = self.eval_obj.eval_x(child1)
            my_Q_y1.append(OFs[0])
            my_Q_y2.append(OFs[1])
            
            if len(my_Q) < my_N * N_c:                     # Crossover the other way if there is still room
                child2 = self.crossover(Mom, Dad)
                child2 = self.mutate(child2)
                my_Q = np.hstack([my_Q, child2])
                OFs = self.eval_obj.eval_x(child2)
                my_Q_y1.append(OFs[0])
                my_Q_y2.append(OFs[1])

        # Convert to numpy arrays
        my_Q_y1 = np.array(my_Q_y1)
        my_Q_y2 = np.array(my_Q_y2)
        
        self.Q_y1 = np.zeros(N, dtype='d')
        self.Q_y2 = np.zeros(N, dtype='d')
        self.Q = np.empty(N*N_c, dtype='d')
        
        # Allgather the x values in Q as well as the objective function evaluations
        self.COMM.Allgather( [my_Q_y1, MPI.DOUBLE], [self.Q_y1, MPI.DOUBLE] )
        self.COMM.Allgather( [my_Q_y2, MPI.DOUBLE], [self.Q_y2, MPI.DOUBLE] )
        self.COMM.Allgather( [my_Q, MPI.DOUBLE], [self.Q, MPI.DOUBLE] )
        
        self.Q = self.Q.reshape([N, N_c])
            
    
    def plot_pop(self, fname = None, gen_num = None):
        
        '''
        Plot the objective function values of each individual in the population
        '''    

        if self.P_y1 is None or self.P_y2 is None:
            raise NameError('Population must be evaluated before plotting.')
        
        plt.plot(self.P_y1, self.P_y2, marker='o', color = 'k', linestyle = 'None')       # population
        plt.xlabel('$y_1$', size=24)
        plt.ylabel('$y_2$', size=24)
            
    #    plt.xlim([-4, 4])
    #    plt.ylim([-4, 4])
            
        
        plt.xticks(size=20)
        plt.yticks(size=20)
        if not gen_num is None:
            plt.title('Generation ' + str(gen_num))
        plt.tight_layout()
        
        if fname is None:
            plt.show()
        else:
            plt.savefig(fname)
            plt.close()