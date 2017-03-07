# main file

import os
import sys
import numpy as np

from AB import AB_model

import networkx as nx
import matplotlib.pyplot as plt


x = AB_model()
x.build_template()
x.generate_defected()
x.template_to_KMC_lattice()
x.show_all()

#G=nx.dodecahedral_graph()
#nx.draw(G)
#plt.draw() 