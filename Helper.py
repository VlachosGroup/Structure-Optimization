# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 17:40:05 2016

@author: mpnun
"""

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