"""
Has classes for various versions of genetic algorithm
"""

import numpy as np
import random
import time
import matplotlib.pyplot as plt
try:
    from mpi4py import MPI
except:
    pass
from NeuralNetwork import NeuralNetwork
    

class MOGA(object):
    
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
        
        self.eval_obj = None                # Must have an eval_x() method which takes a numpy array
        
        self.COMM = None                    # object for MPI parallelization
        
        
    def mutate(self, x, intensity = 1.):
    
        '''
        Mutates an individual to yield an offspring
        '''
        
        x_child = np.array([x[i] for i in range(len(x))])        # copy the data
        
        for i in range(len(x_child)):
            if np.random.random() < intensity / len(x_child):
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
        
        Nc = len(x1)
        index = np.random.randint(0, high = Nc / 2)
        child1 = np.hstack( [x1[:index:], x2[index:index+Nc/2:], x1[index+Nc/2::] ] )
        child2 = np.hstack( [x2[:index:], x1[index:index+Nc/2:], x2[index+Nc/2::] ] )
        
        return [ child1, child2 ]
            

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
            
        self.P_y1 = np.array(self.P_y1)
        self.P_y2 = np.array(self.P_y2)
            
        if n_snaps > 0 and self.COMM.rank == 0:
            snap_ind += 1
            self.plot_pop(fname = 'pop_pic_1.png', gen_num = 0)
            self.eval_obj.show(x = self.P[0,:], n_struc = snap_ind, fmat = 'picture')
    
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
                self.eval_obj.show(x = self.P[0,:], n_struc = snap_ind, fmat = 'picture')
                
        CPU_end = time.time()
        if self.COMM.rank == 0:
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
        
        # Broadcast P_indices and fill self.P, self.P_y1, and self.P_y2 that way
        
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
            children = self.crossover(Dad, Mom)
            n_diffs = np.sum( np.abs( Mom - Dad ) )
            child1 = children[0]
            child2 = children[1]
            child1 = self.mutate(child1, intensity = 1. + 0. * (1. - n_diffs / N_c) )
            my_Q = np.hstack([my_Q, child1])
            OFs = self.eval_obj.eval_x(child1)
            my_Q_y1.append(OFs[0])
            my_Q_y2.append(OFs[1])
            
            if len(my_Q) < my_N * N_c:                     # Crossover the other way if there is still room
                child2 = self.mutate(child2, intensity = 1. + 0. * (1. - n_diffs / N_c) )
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
            
            
class SOGA(MOGA):
    
    '''
    Single-objective genetic algorithm
    Handles one objective function
    '''
    
    def __init__(self):
        
        '''
        Initialize with an empty population
        '''
        
        super(SOGA, self).__init__()
        

    def evolve(self, elite_fraction = 0.5, frac_mutate = 0.0):   
    
        '''
        Execute one generation of the genetic algorithm
        Evolve the population using metation and crossover
        retain: top fraction to keep
        random_select: 
        mutate: 
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
        Assign distance metric for each individual in each front
        '''
        # Extract fitness values for R
        graded_pop = [ [ 0, 0, i ] for i in range( R.shape[0] ) ]
        for i in range( R.shape[0] ):
            graded_pop[i][0] = R_y1[i]
            graded_pop[i][1] = R_y2[i]
            
        graded_pop_arr = np.array( graded_pop )    
        sorted_data = graded_pop_arr[graded_pop_arr[:, 1].argsort()]        # sort according to data in the m'th objective
            
        
        '''
        Selection: Select individuals for the new P
        '''
        
        self.P = []
        self.P_y1 = []
        self.P_y2 = []
            
        ind = 0
        while len(self.P) < N:
        
            self.P.append( R[ sorted_data[ind, -1], : ] )
            self.P_y1.append( R_y1[ int( sorted_data[ind, -1] ) ] )
            self.P_y2.append( R_y2[ int( sorted_data[ind, -1] ) ] )
            
            ind += 1
        
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
            
            contestants = random.sample(range(N), 2)      
            if self.P_y2[contestants[0]] < self.P_y2[contestants[1]]:
                chosen_one = contestants[0]
            else:
                chosen_one = contestants[1]

            new_candidate = self.mutate( self.P[chosen_one,:] )
            my_Q = np.hstack([my_Q, new_candidate])
            if np.array_equal(new_candidate, self.P[chosen_one,:]):      # Mutation may not have changed, so we do not reevaluate the objective functions
                my_Q_y1.append(R_y1[chosen_one])
                my_Q_y2.append(R_y2[chosen_one])
            else:
                OFs = self.eval_obj.eval_x(new_candidate)
                my_Q_y1.append(OFs[0])
                my_Q_y2.append(OFs[1])
          
        # Crossover: Crossover parents to create children
        while len(my_Q) < my_N * N_c:
            
            # Tournament select to choose Mom
            contestants = random.sample(range(N), 2)    
            if self.P_y2[contestants[0]] < self.P_y2[contestants[1]]:
                Mom = self.P[contestants[0],:]
            else:
                Mom = self.P[contestants[1],:]
            
            # Tournament select to choose Dad
            contestants = random.sample(range(N), 2)
            if self.P_y2[contestants[0]] < self.P_y2[contestants[1]]:
                Dad = self.P[contestants[0],:]
            else:
                Dad = self.P[contestants[1],:]
            
            # Crossover Mom and Dad to create a child
            children = self.crossover(Dad, Mom)
            n_diffs = np.sum( np.abs( Mom - Dad ) )
            child1 = children[0]
            child2 = children[1]
            child1 = self.mutate(child1, intensity = 1. + 5. * (1. - n_diffs / N_c) )
            my_Q = np.hstack([my_Q, child1])
            OFs = self.eval_obj.eval_x(child1)
            my_Q_y1.append(OFs[0])
            my_Q_y2.append(OFs[1])
            
            if len(my_Q) < my_N * N_c:                     # Crossover the other way if there is still room
                child2 = self.mutate(child2, intensity = 1. + 5. * (1. - n_diffs / N_c) )
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

        if self.P_y2 is None:
            raise NameError('Population must be evaluated before plotting.')

        plt.hist(self.P_y2, len(self.P_y2)/10)
        plt.xlabel('$y_2$', size=24)
        plt.ylabel('Relative frequency', size=24)    
    #   plt.xlim([-4, 4])
    #   plt.ylim([-4, 4])
            
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
            

class Bifidelity(MOGA):
    
    '''
    Bifidelity genetic algorithm
    '''
    
    def __init__(self):
        
        '''
        Initialize with an empty population
        '''
        
        super(Bifidelity, self).__init__()
        
        self.nn_surrogate = None        # Neural network object used to evaluate an approximate fitness function
        #self.Q = None                   # I don't know why this isn't getting inherited properly
        
        self.evaluated_high_P = None        # Whether each individual in P has been evaluated with the high-fidelity model
        self.evaluated_high_Q = None        # Whether each individual in Q has been evaluated with the high-fidelity model
        self.high_fidelity_evals = 0        # Number of high fidelity evaluations
        self.low_fidelity_evals = 0         # Number of low fidelity evaluations
    
    
    def genetic_algorithm(self, n_gens, frac_controlled = 0.1, n_snaps = 0):
        
        
        '''
        Optimize the population using a genetic algorithm
        '''
        
        
        
        # Set different random seeds for each processor
        random.seed(a=12345)
        np.random.seed(seed=12345)
        
        # Initialize list of steps for taking snapshots
        # The population will be shown after the mutation in the listed generations
        # "After generation 0" gives the initial population
        n_snaps = min([ n_snaps, n_gens + 1])
        snap_ind = 0
        if n_snaps == 0:
            snap_record = []
        elif n_snaps == 1:
            snap_record = [0]
        else:
            snap_record = [int( float(i) / (n_snaps - 1) * ( n_gens ) ) for i in range(n_snaps)]

        # Identify controlled generations
        n_controlled = int(n_gens * frac_controlled)
        control_ind = 0
        if n_controlled == 0:
            controlled_gens = []
        elif n_controlled == 1:
            controlled_gens = [0]
        else:
            controlled_gens = [int( float(i+1) * n_gens / n_controlled ) for i in range(n_controlled)]
            

        # Evaluate individuals in P before the 1st generation
        self.P_y1 = []
        self.P_y2 = []
        for p in self.P:
            OFs = self.eval_obj.eval_x(p)
            self.high_fidelity_evals += 1
            self.P_y1.append(OFs[0])
            self.P_y2.append(OFs[1])
        self.evaluated_high_P = [True for i in range(self.P.shape[0])]
            
        self.P_y1 = np.array(self.P_y1)
        self.P_y2 = np.array(self.P_y2)
        
        # Train neural network on the initial population
        if frac_controlled < 1.:
            control_ind += 1
            self.nn_surrogate = NeuralNetwork()
            self.nn_surrogate.refine(self.P, np.transpose( np.vstack([self.P_y1, self.P_y2]) ) )
            self.nn_surrogate.plot_parity('parity_1.png', title = 'Generation 0')

        # Print population snapshot
        if n_snaps > 0:
            snap_ind += 1
            self.plot_pop(fname = 'pop_pic_1.png', gen_num = 0)
            self.eval_obj.show(x = self.P[0,:], n_struc = snap_ind, fmat = 'picture')
    
        CPU_start = time.time()
        for i in xrange(n_gens):
            
            print 'Generation ' + str(i+1)
                
            # Evolve population
            if i+1 in controlled_gens:
                control_ind += 1
                self.evolve(controlled = True)
                if not self.nn_surrogate is None:
                    self.nn_surrogate.plot_parity(fname = 'parity_' + str(control_ind) +'png', title = 'Generation ' + str(i+1))
            else:
                self.evolve(controlled = False)
            
            # Show the population if it is a snapshot generation
            if i+1 in snap_record:
                snap_ind += 1
                self.plot_pop(fname = 'pop_pic_' + str(snap_ind) + '.png' , gen_num = i+1)
                self.eval_obj.show(x = self.P[0,:], n_struc = snap_ind, fmat = 'picture')
                
        CPU_end = time.time()
        print('Time elapsed: ' + str(CPU_end - CPU_start) + ' seconds')
        print str(self.high_fidelity_evals) + ' high fidelity evaluations.'
        print 'Best individual score: ' + str(self.P_y2[0])
        
    
    def evolve(self, elite_fraction = 0.5, frac_mutate = 0.0, controlled = False):   
    
        '''
        Execute one generation of the genetic algorithm
        Evolve the population using metation and crossover
        retain: top fraction to keep
        random_select: 
        mutate: 
        '''    

        N = self.P.shape[0]         # number of individuals in the population
        N_c = self.P.shape[1]       # number of bits for each individual
        
        # Combine P and Q into R (Q is an empty list in the first generation)
        if self.Q is None:
            R = self.P
            R_y1 = self.P_y1
            R_y2 = self.P_y2
            R_evaluated_high = self.evaluated_high_P
        else:
            R = np.vstack([self.P, self.Q])
            R_y1 = np.hstack([self.P_y1, self.Q_y1])
            R_y2 = np.hstack([self.P_y2, self.Q_y2])
            R_evaluated_high = self.evaluated_high_P + self.evaluated_high_Q

        # Given R, we need to dermine the new P

        '''
        Assign distance metric for each individual in each front
        '''
        # Extract fitness values for R
        graded_pop = [ [ 0, 0, i ] for i in range( R.shape[0] ) ]
        for i in range( R.shape[0] ):
            graded_pop[i][0] = R_y1[i]
            graded_pop[i][1] = R_y2[i]
            
        graded_pop_arr = np.array( graded_pop )    
        sorted_data = graded_pop_arr[graded_pop_arr[:, 1].argsort()]        # sort according to data in the m'th objective
            
        
        '''
        Selection: Select individuals for the new P
        '''
        
        self.P = []
        self.P_y1 = []
        self.P_y2 = []
        self.evaluated_high_P = []
            
        ind = 0

        while len(self.P) < N:
        
            ind_chosen = int(sorted_data[ind, -1])
            self.P.append( R[ ind_chosen, : ] )
            self.P_y1.append( R_y1[ ind_chosen ] )
            self.P_y2.append( R_y2[ ind_chosen ] )
            self.evaluated_high_P.append( R_evaluated_high[ ind_chosen ] )
            
            ind += 1
        
        self.P = np.array(self.P)
        self.P_y1 = np.array(self.P_y1)
        self.P_y2 = np.array(self.P_y2)
        
        '''
        Given the new P, use tournament selection, mutation and crossover to create Q
        '''        
        
        my_N = N
        my_Q = np.array([])
        my_Q_y1 = []
        my_Q_y2 = []

        # Mutation: Tournament select parents and mutate to create children
        while len(my_Q) < int(my_N * frac_mutate) * N_c:
            
            contestants = random.sample(range(N), 2)      
            if self.P_y2[contestants[0]] < self.P_y2[contestants[1]]:
                chosen_one = contestants[0]
            else:
                chosen_one = contestants[1]

            new_candidate = self.mutate( self.P[chosen_one,:] )
            my_Q = np.hstack([my_Q, new_candidate])
            if np.array_equal(new_candidate, self.P[chosen_one,:]):      # Mutation may not have changed, so we do not reevaluate the objective functions
                my_Q_y1.append(R_y1[chosen_one])
                my_Q_y2.append(R_y2[chosen_one])
            else:
                if controlled:
                    OFs = self.eval_obj.eval_x(new_candidate)
                    self.high_fidelity_evals += 1
                    my_Q_y1.append(OFs[0])
                    my_Q_y2.append(OFs[1])
                else:
                    OFs = self.nn_surrogate.predict(new_candidate.reshape(1,-1))
                    my_Q_y1.append(OFs[0,0])
                    my_Q_y2.append(OFs[0,1])
          
        # Crossover: Crossover parents to create children
        while len(my_Q) < my_N * N_c:
            
            # Tournament select to choose Mom
            contestants = random.sample(range(N), 2)    
            if self.P_y2[contestants[0]] < self.P_y2[contestants[1]]:
                Mom = self.P[contestants[0],:]
            else:
                Mom = self.P[contestants[1],:]
            
            # Tournament select to choose Dad
            contestants = random.sample(range(N), 2)
            if self.P_y2[contestants[0]] < self.P_y2[contestants[1]]:
                Dad = self.P[contestants[0],:]
            else:
                Dad = self.P[contestants[1],:]
            
            # Crossover Mom and Dad to create a child
            children = self.crossover(Dad, Mom)
            n_diffs = np.sum( np.abs( Mom - Dad ) )
            child1 = children[0]
            child2 = children[1]
            child1 = self.mutate(child1, intensity = 1. + 5. * (1. - n_diffs / N_c) )
            my_Q = np.hstack([my_Q, child1])
            if controlled:
                OFs = self.eval_obj.eval_x(child1)
                self.high_fidelity_evals += 1
                my_Q_y1.append(OFs[0])
                my_Q_y2.append(OFs[1])
            else:
                OFs = self.nn_surrogate.predict(child1.reshape(1,-1))
                my_Q_y1.append(OFs[0,0])
                my_Q_y2.append(OFs[0,1])
            
            if len(my_Q) < my_N * N_c:                     # Crossover the other way if there is still room
                child2 = self.mutate(child2, intensity = 1. + 5. * (1. - n_diffs / N_c) )
                my_Q = np.hstack([my_Q, child2])
                if controlled:
                    OFs = self.eval_obj.eval_x(child1)
                    self.high_fidelity_evals += 1
                    my_Q_y1.append(OFs[0])
                    my_Q_y2.append(OFs[1])
                else:
                    OFs = self.nn_surrogate.predict(child2.reshape(1,-1))
                    my_Q_y1.append(OFs[0,0])
                    my_Q_y2.append(OFs[0,1])

        # Convert to numpy arrays
        my_Q_y1 = np.array(my_Q_y1)
        my_Q_y2 = np.array(my_Q_y2)
        
        # Allgather the x values in Q as well as the objective function evaluations
        self.Q_y1 = my_Q_y1
        self.Q_y2 = my_Q_y2
        self.Q = my_Q
        
        self.Q = self.Q.reshape([N, N_c])
        
        # If it is a controlled generation, evaluate P and Q with the full model
        # Add their data to refine the neural network
        if controlled:
            
            self.evaluated_high_Q = [True for i in xrange(N)]
            
            # Evaluate individuals in P with the full model
            for indiv in xrange(N):
                if not self.evaluated_high_P[indiv]:
                    OFs = self.eval_obj.eval_x(self.P[indiv,:])
                    self.P_y1[indiv] = OFs[0]
                    self.P_y2[indiv] = OFs[1]
                    self.evaluated_high_P[indiv] = True           
            
            # Refine surrogate model
            if not self.nn_surrogate is None:
                # Prepare new training data
                newX = np.vstack([self.P, self.Q])
                newY1 = np.hstack([self.P_y1, self.Q_y1])
                newY2 = np.hstack([self.P_y2, self.Q_y2])
                newY = np.transpose( np.vstack( [newY1, newY2] ) )
                
                self.nn_surrogate.refine(newX, newY )
                
        else:
            self.evaluated_high_Q = [False for i in xrange(N)]
        
        
    def plot_pop(self, fname = None, gen_num = None):
        
        '''
        Plot the objective function values of each individual in the population
        '''    

        if self.P_y2 is None:
            raise NameError('Population must be evaluated before plotting.')

        plt.hist(self.P_y2, len(self.P_y2)/10)
        plt.xlabel('$y_2$', size=24)
        plt.ylabel('Relative frequency', size=24)    
    #   plt.xlim([-4, 4])
    #   plt.ylim([-4, 4])
            
        plt.xticks(size=20)
        plt.yticks(size=20)
        plt.xlim([-90,0])
        plt.ylim([0,60])
        if not gen_num is None:
            plt.title('Generation ' + str(gen_num))
        plt.tight_layout()
        
        if fname is None:
            plt.show()
        else:
            plt.savefig(fname)
            plt.close()