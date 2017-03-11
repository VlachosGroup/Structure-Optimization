# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 17:40:05 2016

@author: mpnun
"""

import numpy as np
from ase import Atoms

class Helper:
    
    @staticmethod
    def sort_atoms(mol_dat):
        
        new_mol = Atoms()
        new_mol.cell = mol_dat.cell
        sorted_atom_inds = sorted(range(len(mol_dat.arrays['numbers'])), key=lambda k: mol_dat.arrays['numbers'][k])
        
        for atom_ind in sorted_atom_inds:
            atom_to_add = Atoms(mol_dat.get_chemical_symbols()[atom_ind], positions = [mol_dat.positions[atom_ind,:]])
            new_mol.extend(atom_to_add)
        
        return new_mol
        
    @staticmethod
    def find_bridge(mol_dat, cutoff = 1.6, periodicity = [True, False, False]):
        
        # Set up symmetry cells
        x_cells = [0]
        y_cells = [0]
        z_cells = [0]        
        if periodicity[0]:
            x_cells = [-1,0,1]
        if periodicity[1]:
            y_cells = [-1,0,1]
        if periodicity[2]:
            z_cells = [-1,0,1]
        
        offsets = []
        for x_cell in x_cells:
            for y_cell in y_cells:
                for z_cell in z_cells:
                    offsets.append([x_cell, y_cell, z_cell])

        # Build bridge site objects
        bridge_sites = Atoms()
        bridge_sites.cell = mol_dat.cell
        
        for atom_ind_1 in range(len(mol_dat.arrays['numbers'])):
            for atom_ind_2 in range(atom_ind_1):
                
                # Test each periodic image of the second atom
                for offset in offsets:
                    loc2 = np.dot(offset,bridge_sites.cell) + mol_dat.positions[atom_ind_2]
                
                    dist = np.linalg.norm(mol_dat.positions[atom_ind_1] - loc2)
                    if dist < cutoff:
                        bridge_pos = (mol_dat.positions[atom_ind_1] + loc2) / 2
                        bridge_atom = Atoms('Xe', positions=[bridge_pos])
                        bridge_sites.extend(bridge_atom)
        
        return bridge_sites