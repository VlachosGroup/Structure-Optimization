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
        
        bridge_sites = Atoms()
        bridge_sites.cell = mol_dat.cell
        
        for atom_ind_1 in range(len(mol_dat.arrays['numbers'])):
            for atom_ind_2 in range(atom_ind_1):
                dist = np.linalg.norm(mol_dat.positions[atom_ind_1] - mol_dat.positions[atom_ind_2])
                if dist < cutoff:
                    bridge_pos = (mol_dat.positions[atom_ind_1] + mol_dat.positions[atom_ind_2]) / 2
                    bridge_atom = Atoms('S', positions=[bridge_pos])
                    bridge_sites.extend(bridge_atom)
        
        return bridge_sites