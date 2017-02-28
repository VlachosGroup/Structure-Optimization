import sys
import os
import numpy as np
import matplotlib as mat
import matplotlib.pyplot as plt

from ase.build import fcc111
from ase.io import read
from ase.visualize import view
from ase.io import write

sys.path.append('/home/vlachos/mpnunez/Github/Zacros-Wrapper/PythonCode')
import Core.Lattice as lat

''' Create ASE atoms object '''

slab = fcc111('Pt', size=(8, 8, 5), vacuum=15.0)

coords = slab.get_positions()
a_nums = slab.get_atomic_numbers()
chem_symbs = slab.get_chemical_symbols()
n_atoms = len(chem_symbs)
xyz_cell = slab.get_cell()

# Change top layer atoms to Ni
top_layer_z = np.max(coords[:,2])
for atom_ind in range(n_atoms):
    if coords[atom_ind,2] > top_layer_z - 0.1:
        a_nums[atom_ind] = 28
        chem_symbs[atom_ind] = 'Ni'
        
slab.set_atomic_numbers(a_nums)
slab.set_chemical_symbols(chem_symbs)

# Extract fractional coordinates
frac_coords = np.dot(slab.get_positions(), np.linalg.inv(xyz_cell))

''' Make structure into a KMC lattice '''

slab_lat = lat()

slab_lat.workingdir = '.'
slab_lat.lattice_matrix = xyz_cell[0:2, 0:2]
slab_lat.repeat = [1,1]
slab_lat.site_type_names = ['Pt', 'Ni']
slab_lat.site_type_inds = [1 for i in range(n_atoms)]

for i in range(n_atoms):
    if chem_symbs[i] == 'Pt':
        slab_lat.site_type_inds[i] = 1
    elif chem_symbs[i] == 'Ni':
        slab_lat.site_type_inds[i] = 2
    else:
        raise ValueError('Unknown atom type')

slab_lat.frac_coords = frac_coords[:,0:2]
slab_lat.cart_coords = coords
slab_lat.Build_neighbor_list()

''' Show results '''

# View slab in ASE GUI
#view(slab)
write('ase_slab.png', slab)

# Write lattice_input.dat
slab_lat.Write_lattice_input()

# Graph the lattice
plt = slab_lat.PlotLattice(plot_neighbs = True)
plt.savefig(os.path.join('.', 'lattice.png'))
plt.close()