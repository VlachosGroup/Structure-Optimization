# -*- coding: utf-8 -*-
"""
Created on Fri May 26 13:41:45 2017

@author: mpnun
"""

import numpy as np
import copy
import random
import os
import matplotlib.pyplot as plt
import matplotlib as mat
import matplotlib.ticker as mtick
import time

from ase.build import fcc111, fcc100
from ase.neighborlist import NeighborList
from ase.io import write

from metal import metal
from ORR import ORR_rate
from graph_theory import Graph

class cat_structure:
    
    '''
    Catalyst structure with defects
    '''    
    
    def __init__(self, met_name, facet, dim1, dim2):
        
        metal = None
        atoms_obj_template = None
        active_atoms = None
        variable_atoms = None
        template_graph = None
        defected_graph = None
        surface_area = None                 # surface area of the slab in square angstroms
        
        '''
        Build the slab
        '''        
        
        self.metal = metal(met_name)
        
        if facet == '111' or facet == 111:
            self.atoms_obj_template = fcc111(met_name, size=(dim1, dim2, 4), vacuum=15.0)
        elif facet == '100' or facet == 100:
            self.atoms_obj_template = fcc100(met_name, size=(dim1, dim2, 4), vacuum=15.0)
        else:
            raise ValueError(str(facet) + ' is not a valid facet.')
            
        self.atoms_obj_template.set_pbc([True, True, False])
        
        # Find neighbors based on distances
        rad_list = ( 2.77 + 0.2 ) / 2 * np.ones(len(self.atoms_obj_template))               # list of neighboradii for each site
        neighb_list = NeighborList(rad_list, self_interaction = False)      # set bothways = True to include both ways
        neighb_list.build(self.atoms_obj_template)
        
        self.template_graph = Graph()
        for i in range(len(self.atoms_obj_template)):
            self.template_graph.add_vertex(i)
        
        for i in range(len(self.atoms_obj_template)):
            for j in neighb_list.neighbors[i]:
                self.template_graph.add_edge([i,j])
            
        self.active_atoms = range(2 * dim1 * dim2, 4 * dim1 * dim2)
        self.variable_atoms = range(3 * dim1 * dim2, 4 * dim1 * dim2)
        self.defected_graph = copy.deepcopy(self.template_graph)        
        
        # Compute surface area for use in normalization      
        self.surface_area = np.linalg.norm( np.cross( self.atoms_obj_template.get_cell()[0,:], self.atoms_obj_template.get_cell()[1,:] ) )        

    
    def eval_current_density(self, atom_graph = None, normalize = False):
        
        '''
        Normalized: current density [mA/cm^2]
        Not normalized: current [mA]
        '''

        if atom_graph is None:
            atom_graph = self.defected_graph

        curr = 0
        for i in self.active_atoms:
            if atom_graph.is_node(i):
                if atom_graph.get_coordination_number(i) <= 9:
                    gcn = atom_graph.get_generalized_coordination_number(i, 12)
                    BE_OH = self.metal.get_OH_BE(gcn)
                    BE_OOH = self.metal.get_OOH_BE(gcn)
                    curr += ORR_rate(BE_OH, BE_OOH)
                
        if normalize:
            pass
#            curr = curr / SA           # normalize by surface area
            
        return curr
        
    
    def eval_surface_energy(self, atom_graph = None, normalize = False):
        
        '''
        Normalized: surface energy [J/m^2]
        Not normalized: formation energy [eV]
        '''        
        
        if atom_graph is None:
            atom_graph = self.defected_graph
        
        E_form = 0
        for i in self.active_atoms:
            if atom_graph.is_node(i):
                E_form += self.metal.E_coh * ( 1 - np.sqrt( atom_graph.get_coordination_number(i) / 12.0 ) )
                
        if normalize:
            pass
#            E_form = E_form / SA           # normalize by surface area
                
        return E_form

        
    def seed_occs(self, coverage = 0.5):
        
        '''
        Randomize the occupancies in the top layer
        '''
        
        n_vacancies = round( coverage * len(self.variable_atoms) )
        n_vacancies = int(n_vacancies)
        vacant_sites = random.sample(self.variable_atoms, n_vacancies)
        for i in vacant_sites:
            self.defected_graph.remove_vertex(i)
        
        
    def get_defected_mols(self):
        
        '''
        Use defected graph and template atoms object to generate
        the atoms object for the defected structure
        '''
        
        atoms_obj = copy.deepcopy(self.atoms_obj_template)
        delete_these = [False for i in range(len(atoms_obj))]
        for i in self.variable_atoms:
            if not self.defected_graph.is_node(i):
                delete_these[i] = True
        
        delete_these = np.array(delete_these)
        del atoms_obj[delete_these]        
        return atoms_obj
        
    
    def flip_atom(self, ind):
        
        '''
        If atom number ind is present in the defected graph, remove it.
        If it is not present, add it and all edges to adjacent atoms.
        '''        
        
        if self.defected_graph.is_node(ind):
            self.defected_graph.remove_vertex(ind)
        else:
            self.defected_graph.add_vertex(ind)
            for neighb in self.template_graph.get_neighbors(ind):
                if self.defected_graph.is_node(neighb):
                    self.defected_graph.add_edge([ind, neighb])

        
    def optimize(self, ensemble = 'GCE', omega = 1, n_cycles = 1, n_record = 26, n_snaps = 0):
        
        '''
        Use simulated annealing to optimize defects on the surface
        ensemble: CE (canonical ensemble) or GCE (grand canonical ensemble)
        omega: Pareto coefficient, 0 optimizes surface energy, 1 optimized activity
        n_record: number of data snapshots to save in a trajectory
        n_snaps: number of snapshots to draw
        '''
        
        # Need a variable to control the cooling schedule

        total_steps = n_cycles * len(self.variable_atoms)
        
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
        

        # Evaluate initial structure
        #pure_current = self.eval_current_density(atom_graph = self.template_graph)
        #pure_E_form = self.eval_surface_energy(atom_graph = self.template_graph)
        E_form_norm = self.metal.E_coh / 12.0
        
        BE_OH_ideal = self.metal.get_OH_BE(8.3)
        BE_OOH_ideal = self.metal.get_OOH_BE(8.3)
        curr_norm = - 0.25 * ORR_rate(BE_OH_ideal, BE_OOH_ideal)
        
        # Evaluate initial structure
        current = self.eval_current_density()
        E_form = self.eval_surface_energy()
        OF = (1.0 - omega) / E_form_norm * E_form + omega / curr_norm * current
        
        # Record initial data
        surf_eng_traj[0] = E_form * 1.60218e-19 / ( self.surface_area * 10e-20)  # convert eV to Joule per m^2
        current_traj[0] = current / ( self.surface_area * 10e-16)        # convert from mA to mA / cm^2
        obj_func_traj[0] = OF
        record_ind += 1        
        
        # Snapshot the initial state
        self.record_snapshot(snap_ind, record_ind, step_record, surf_eng_traj, current_traj)
        snap_ind += 1        
        
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
                atom_to_flip = random.choice(self.variable_atoms)
                self.flip_atom(atom_to_flip)
            elif ensemble == 'CE':
                
                
                good_site = False
                while not good_site:
                    move_from = random.choice(self.variable_atoms)              # Choose random atoms until you find one with missing neighbors
                    if self.defected_graph.is_node(move_from):
                        if self.defected_graph.get_coordination_number(move_from) < 9:
                            good_site = True
                    
                present_neighbs = self.defected_graph.get_neighbors(move_from)
                all_neighbs = self.template_graph.get_neighbors(move_from)
                vac_list = []
                for site in all_neighbs:
                    if not site in present_neighbs:
                        vac_list.append(site)
                
                move_to = random.choice(vac_list)
                
                self.flip_atom(move_from)
                self.flip_atom(move_to)
                
            else:
                raise ValueError(ensemble + ' is not a valid ensemble.')
                
            # Evaluate the new structure and determine whether or not to accept
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
                    self.flip_atom(move_from)
                    self.flip_atom(move_to)
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
                surf_eng_traj[record_ind] = E_form * 1.60218e-19 / ( self.surface_area * 10e-20 )  # convert eV to Joule per m^2
                current_traj[record_ind] = current / ( self.surface_area * 10e-16 )        # convert from mA to mA / cm^2
                obj_func_traj[record_ind] = OF
                record_ind += 1
            
            # Record snapshot
            if step+1 in snap_record:
                self.record_snapshot(snap_ind, record_ind, step_record, surf_eng_traj, current_traj)
                snap_ind += 1
                print 'Printing snapshot ' + str(snap_ind)
                
        CPU_end = time.time()
        print('Time elapsed: ' + str(CPU_end - CPU_start) )
                
                
    def record_snapshot(self, snap_ind, record_ind, step_record, surf_eng_traj, current_traj):
        
        '''
        Draw a double y-axis graph of current density and surface energy vs. optimiztion step
        '''        
        
        fudge_factor = 10.0     # My surface energies and current densities are off by a factor of 10 for some reason. Need to debug this...        
        
        mat.rcParams['mathtext.default'] = 'regular'
        mat.rcParams['text.latex.unicode'] = 'False'
        mat.rcParams['legend.numpoints'] = 1
        mat.rcParams['lines.linewidth'] = 2
        mat.rcParams['lines.markersize'] = 12
#        mat.rc('xtick', labelsize=20)
        mat.rc('ytick', labelsize=20) 
        
        fig, ax1 = plt.subplots()
        ax1.plot(step_record[0:record_ind:], surf_eng_traj[0:record_ind:] * fudge_factor, 'b-')
        ax1.set_xlabel('Metropolis step', size=24)
        # Make the y-axis label, ticks and tick labels match the line color.
        ax1.set_ylabel(r'Surface energy, $\gamma$ (J/m$^2$)', color='b', size=24)
        ax1.tick_params('y', colors='b', size = 20)
        plt.xticks( [ step_record[0], step_record[-1] / 4, step_record[-1] / 2, int(0.75 * step_record[-1]), step_record[-1] ] )
        ax1.set_xlim( [ step_record[0], step_record[-1] ] )
#        ax1.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.0e'))
#        plt.locator_params(axis='x', nticks=5)
        ax1.set_ylim([2.0, 2.4])            # arbitrary bounds set for GIF
        
        ax2 = ax1.twinx()
        ax2.set_xlim( [ step_record[0], step_record[-1] ] )
        ax2.plot(step_record[0:record_ind:], current_traj[0:record_ind:] * fudge_factor, 'r-')
        ax2.set_ylabel(r'Current density, $j$ (mA/cm$^2$)', color='r', size=24)
        ax2.tick_params('y', colors='r', size=20)
        ax2.set_ylim([0, 100])                # arbitrary bounds set for GIF
        
        out_fldr = 'C:\Users\mpnun\Dropbox\ProfDev\Interview\Intel\Slides\ORR\opt_movie'
        fig.tight_layout()
        plt.savefig(os.path.join(out_fldr, 'opt_graph_' + str(snap_ind) + '.png' ))
        plt.close()
                
        '''
        Print image of surface
        '''        
        
        defect_atoms_obj = self.get_defected_mols()
        
        coords = defect_atoms_obj.get_positions()
        a_nums = defect_atoms_obj.get_atomic_numbers()
        chem_symbs = defect_atoms_obj.get_chemical_symbols()
        
        # Change top layer atoms to Ni
        top_layer_z = np.max(coords[:,2])
        for atom_ind in range(len(defect_atoms_obj)):
            if coords[atom_ind,2] > top_layer_z - 0.1:
                a_nums[atom_ind] = 27
                chem_symbs[atom_ind] = 'Co'
                
        defect_atoms_obj.set_atomic_numbers(a_nums)
        defect_atoms_obj.set_chemical_symbols(chem_symbs)
        
        write(os.path.join(out_fldr, 'image_defect_' + str(snap_ind) + '.png' ), defect_atoms_obj )