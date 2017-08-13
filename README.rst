Catalyst Structure Optimization
================================

Catalyst structure optimization in Python. Sphinx documentation is contained in the docs folder.
It is applied to 

* NH3 decomposition on NiPt
* Oxygen reduction reaction on Pt and Au

Dependencies
-------------
* `Atomic simualtion environment <https://wiki.fysik.dtu.dk/ase/>`_ : Data structures for molecular structures anf file IO.
* `NetworkX <http://networkx.github.io/index.html>`_ : Handles graph theroy tasks such as node attributes and subgraph isomorphisms.
* `Zacros-Wrapper <https://github.com/VlachosGroup/Zacros-Wrapper>`_ : Used to build the lattice and run KMC simulations for NH3 decomposition.
* `mpi4py <http://pythonhosted.org/mpi4py/>`_ : Used to parallelize the genetic algorithm

Publications
-------------
* M. Nunez, J. Lansford, D.G. Vlachos, "Optimization of transition metal catalyst facet structure: Application to the oxygen reduction reaction" (under review)
* M. Nunez, G. Gu., D.G. Vlachos, "Catalyst Structure Optimization using Data Driven Model Reduction" (in preparation)

Developers
-----------
Marcel Nunez (mpnunez@udel.edu)