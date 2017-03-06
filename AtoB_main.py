# main file

import os
import sys
import numpy as np

sys.path.append('/home/vlachos/mpnunez/Github/Zacros-Wrapper/PythonCode')
import Core as zw

sys.path.append('/home/vlachos/mpnunez/Github/networkx')
import networkx as nx

class Cat_structure:
    
    def __init__(self):
        
        pass
        
    
    def randomize_occs(self):
    
        pass
        

if __name__ == "__main__":
        
    G=nx.Graph()
    G.add_edge(1,2) # default edge data=1
    G.add_edge(2,3,weight=0.9) # specify edge data
    
    print 'Hi Marcel!'