'''
Has template class to be inherited from for specific models
'''

import numpy as np
import os
import sys
import copy
import random
import time
import matplotlib.pyplot as plt
import matplotlib as mat
import matplotlib.ticker as mtick

#sys.path.append('/home/vlachos/mpnunez/ase')
sys.path.append('C:\Users\mpnun\Dropbox\Coding\Python_packages\ase')
from ase.io import read
from ase.io import write

sys.path.append('C:\Users\mpnun\Dropbox\Coding\Python_packages\networkx')
import networkx as nx
import networkx.algorithms.isomorphism as iso

#sys.path.append('/home/vlachos/mpnunez/Github/Zacros-Wrapper')
sys.path.append('C:\Users\mpnun\Dropbox\Github\Zacros-Wrapper')
import zacros_wrapper.Lattice as lat


class dyno_struc(object):
    
    '''
    Handles a dynamic structure upon which graphs and KMC lattices can be built
    '''
    
    def __init__(self):
    
        '''
        Inherit from object class and declare up class variables
        '''
    
        super(dyno_struc, self).__init__()
        
        self.path = '.'        
        
        self.ASE_template = None            # will be an ASE atoms object when
        self.variable_atoms = None                   # indices of the atoms which can be changed
        
        # Data structures for the defected structure
        self.atoms_missing = None               # atoms_missing of atoms in the ASE atoms object, True if it is changed relative to the template
        self.ASE_defected = None            # ASE atoms object, with some atoms removed or transmuted
        self.molecular_NetX = None                 # networkx graph object
        self.KMC_lat = None                   # KMC lattice object
        self.KMC_NetX = None                 # KMC lattice as a NetworkX graph
        
        # Input to neural network
        self.target_graph = None        # networkx graph object, subgraphs to
        self.target_isos = None        # list of isomorphisms
        self.target_mult = None        # networkx graph object, subgraphs to
        
    
    def ASE_to_atoms_missing(self):
    
        '''
        Determine which atoms in the template are missing in the defected structure
        '''
    
        self.atoms_missing = np.array([False for i in range(len(self.ASE_template))])   # Need to find which atoms are missing or changed in the defected structure
        
        d_cut = 0.001        # distance cutoff in angstroms
        for atom_ind in range(len(self.ASE_template)):
        
            # Get position and atomic number of the template atom we are trying to find
            cart_coords = self.ASE_template.get_positions()[atom_ind, :]
            atomic_num = self.ASE_template.get_atomic_numbers()[atom_ind]
            
            defect_ind = 0      # index of defect atom which might be a match
            dist = 1.0      # distance between atoms we are trying to match
            
            match_found = False
            
            while (not match_found) and defect_ind < len(self.ASE_defected):
            
                defect_coords = self.ASE_defected.get_positions()[defect_ind, :]
                defect_an = self.ASE_defected.get_atomic_numbers()[defect_ind]
                dist = np.linalg.norm( cart_coords - defect_coords )
                match_found = (dist < d_cut) and (defect_an == atomic_num)
                defect_ind += 1
                
            if not match_found:
                self.atoms_missing[atom_ind] = True
    

    def atoms_missing_to_ASE(self):
        
        '''
        Use defected graph and template atoms object to generate
        the atoms object for the defected structure
        '''
        
        self.ASE_defected = copy.deepcopy(self.ASE_template)
        del self.ASE_defected[self.atoms_missing]        
        
     
    def KMC_lattice_to_graph2D(self):
    
        '''
        Convert KMC lattice object to a networkx graph object
        '''
        
        self.KMC_NetX = nx.Graph()
        self.KMC_NetX.add_nodes_from(range(len(kmc_lat.site_type_inds)))
        self.KMC_NetX.add_edges_from(kmc_lat.neighbor_list)

        for i in range(len(kmc_lat.site_type_inds)):
            self.KMC_NetX.node[i]['type'] = kmc_lat.site_type_names[kmc_lat.site_type_inds[i]-1]
            

    def count_fingerprints(self):
        
        '''
        Enumerate subgraph isomorphisms
        '''
        
        n_fingerprints = len(self.fingerprint_graphs)
        self.fingerprint_counts = [0 for j in range(n_fingerprints)]
        
        for i in range(n_fingerprints):
            
            # Count subgraphs
            GM = iso.GraphMatcher(self.molecular_NetX, self.fingerprint_graphs[i], node_match=iso.categorical_node_match('type','Au'))
            
            n_subgraph_isos = 0
            for subgraph in GM.subgraph_isomorphisms_iter():
                n_subgraph_isos += 1
            
            # Count symmetry of the subgraph
            GM_sub = iso.GraphMatcher(self.fingerprint_graphs[i], self.fingerprint_graphs[i], node_match=iso.categorical_node_match('type','Au'))            
            
            symmetry_count = 0
            for subgraph in GM_sub.subgraph_isomorphisms_iter():
                symmetry_count += 1
                
            self.fingerprint_counts[i] = n_subgraph_isos / symmetry_count
            

    def count_target(self):
        
        '''
        Find all subgrpah isomorphisms of the target graph in the molecular graph
        '''        
 
        if self.target_graph is None:
            raise NameError('Target graph not defined.')
            
        # Fill a list with the isomorphisms
        self.target_isos = []
        GM = iso.GraphMatcher(self.molecular_NetX, self.target_graph, node_match=iso.categorical_node_match('element','Au'))
        for subgraph in GM.subgraph_isomorphisms_iter():
            self.target_isos.append(subgraph)
        
        # Count symmetry of the subgraph
        self.target_mult = 0
        GM_sub = iso.GraphMatcher(self.target_graph, self.target_graph, node_match=iso.categorical_node_match('element','Au'))            
        for subgraph in GM_sub.subgraph_isomorphisms_iter():
            self.target_mult += 1
            
        # Compute the diameter of the target graph
        
        
    def optimize(self, ensemble = 'GCE', omega = 1, n_cycles = 1, n_record = 100, n_snaps = 0):
        
        '''
        Use simulated annealing to optimize defects on the surface
        ensemble: CE (canonical ensemble) or GCE (grand canonical ensemble)
        omega: Pareto coefficient, 0 optimizes surface energy, 1 optimized activity
        n_record: number of data snapshots to save in a trajectory
        n_snaps: number of snapshots to draw
        '''
        
        # Need a variable to control the cooling schedule

        total_steps = n_cycles * len(self.ASE_template)
        
        # Initialize variables to record the trajectory
        step_record = [int( float(i) / (n_record - 1) * ( total_steps ) ) for i in range(n_record)]
        surf_eng_traj = np.zeros(n_record)        
        current_traj = np.zeros(n_record)
        obj_func_traj = np.zeros(n_record)
        record_ind = 0
        snap_ind = 0
        
        # Initialize list of steps for taking snapshots
        if n_snaps == 0:
            snap_record = []
        elif n_snaps == 1:
            snap_record = [0]
        else:
            snap_record = [int( float(i) / (n_snaps - 1) * ( total_steps ) ) for i in range(n_snaps)]
        
        E_form_norm = 1.0
        curr_norm = -1.0
        
        # Evaluate initial structure
        self.count_target()
        current = self.eval_current_density()
        E_form = self.eval_surface_energy()
        OF = (1.0 - omega) / E_form_norm * E_form + omega / curr_norm * current
        
        # Record initial data
        surf_eng_traj[0] = E_form
        current_traj[0] = current
        obj_func_traj[0] = OF
        record_ind += 1        
        
        # Snapshot the initial state
        snap_ind += 1
        self.record_snapshot(snap_ind, record_ind, step_record, surf_eng_traj, current_traj)  
        print 'Printing snapshot ' + str(snap_ind)
        
        CPU_start = time.time()        
        
        for step in range( total_steps ):
            
            # Set temperature
            Metro_temp = 0.7 * (1 - float(step) / total_steps)

            # Possibly change ensemble and activty-weight for quenching
            if step > int(total_steps * 0.95):
                omega = 0
                Metro_temp = 0
                ensemble = 'CE'
            
            # Record data before changing structure
            current_prev = current
            E_form_prev = E_form
            OF_prev = OF
            
            # Do a Metropolis move
            if ensemble == 'GCE':
            
                atom_to_flip = self.get_GCE_ind()
                self.flip_atom(atom_to_flip)
                
            elif ensemble == 'CE':
                
                to_from = self.get_CE_inds()
                self.flip_atom(to_from[0])
                self.flip_atom(to_from[1])
                
            else:
                raise ValueError(ensemble + ' is not a valid ensemble.')
                
            '''
            Evaluate the new structure and determine whether or not to accept
            '''
                
            current = self.eval_current_density()       
            E_form = self.eval_surface_energy()
            OF = (1.0 - omega) / E_form_norm * E_form + omega / curr_norm * current
            
            if OF - OF_prev < 0:                # Downhill move
                accept = True
            else:                               # Uphill move
                if Metro_temp > 0:              # Finite temperature, may accept uphill moves
                    accept = np.exp( - ( OF - OF_prev ) / Metro_temp ) > random.random()
                else:                           # Zero temperature, never accept uphill moves
                    accept = False
            
            # Reverse the change if the move is not accepted
            if not accept:
                
                # Revert the change
                if ensemble == 'GCE':
                    self.flip_atom(atom_to_flip)
                elif ensemble == 'CE':
                    self.flip_atom(to_from[0])
                    self.flip_atom(to_from[1])
                else:
                    raise ValueError(ensemble + ' is not a valid ensemble.')
                
                # Use previous values for evaluations
                current = current_prev
                E_form = E_form_prev
                OF = OF_prev
            
            '''
            Record data and snapshots 
            '''            
            
            # Record data
            if step+1 in step_record:
                surf_eng_traj[record_ind] = E_form
                current_traj[record_ind] = current
                obj_func_traj[record_ind] = OF
                record_ind += 1
            
            # Record snapshot
            if step+1 in snap_record:
                snap_ind += 1
                self.record_snapshot(snap_ind, record_ind, step_record, surf_eng_traj, current_traj)                
                print 'Printing snapshot ' + str(snap_ind)
                
        CPU_end = time.time()
        print('Time elapsed: ' + str(CPU_end - CPU_start) )
        
        
    def record_snapshot(self, snap_ind, record_ind, step_record, surf_eng_traj, current_traj):
        
        '''
        Draw a double y-axis graph of current density and surface energy vs. optimiztion step
        '''        
              
        
        mat.rcParams['mathtext.default'] = 'regular'
        mat.rcParams['text.latex.unicode'] = 'False'
        mat.rcParams['legend.numpoints'] = 1
        mat.rcParams['lines.linewidth'] = 2
        mat.rcParams['lines.markersize'] = 12
#        mat.rc('xtick', labelsize=20)
        mat.rc('ytick', labelsize=20) 
        
        fig, ax1 = plt.subplots()
        ax1.plot(step_record[0:record_ind:], surf_eng_traj[0:record_ind:], 'b-')
        ax1.set_xlabel('Metropolis step', size=24)
        # Make the y-axis label, ticks and tick labels match the line color.
        ax1.set_ylabel(r'Surface energy, $\gamma$ (J/m$^2$)', color='b', size=24)
        ax1.tick_params('y', colors='b', size = 20)
        plt.xticks( [ step_record[0], step_record[-1] / 4, step_record[-1] / 2, int(0.75 * step_record[-1]), step_record[-1] ] )
        ax1.set_xlim( [ step_record[0], step_record[-1] ] )
        #ax1.set_ylim([2.0, 2.4])            # arbitrary bounds set for GIF
        
        ax2 = ax1.twinx()
        ax2.set_xlim( [ step_record[0], step_record[-1] ] )
        ax2.plot(step_record[0:record_ind:], current_traj[0:record_ind:], 'r-')
        ax2.set_ylabel(r'Current density, $j$ (mA/cm$^2$)', color='r', size=24)
        ax2.tick_params('y', colors='r', size=20)
        #ax2.set_ylim([0, 100])                # arbitrary bounds set for GIF
        
        out_fldr = 'opt_movie'
        fig.tight_layout()
        plt.savefig(os.path.join(out_fldr, 'opt_graph_' + str(snap_ind) + '.png' ))
        plt.close()
                
        '''
        Print image of surface
        '''        
        
        self.atoms_missing_to_ASE()
        write(os.path.join(out_fldr, 'image_defect_' + str(snap_ind) + '.png' ), self.ASE_defected )