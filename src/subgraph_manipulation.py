# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 12:40:38 2017

@author: mpnun
"""

'''
Test graph manipulation with subgraphs
'''

import networkx as nx
import networkx.algorithms.isomorphism as iso


'''
Create big graph
'''

big_graph = nx.Graph()
big_graph.add_nodes_from(['A', 'B', 'C', 'D'])
big_graph.add_edges_from([['A', 'B'], ['A', 'C'], ['A', 'D'], ['B', 'C'], ['B', 'D'], ['C', 'D']])

'''
Create small graph
'''

small_graph = nx.Graph()
small_graph.add_nodes_from(['E', 'F'])
small_graph.add_edges_from([['E', 'F']])

'''
Count subgraphs
'''

GM = iso.GraphMatcher(big_graph, small_graph)
subgraph_list = []
for subgraph in GM.subgraph_isomorphisms_iter():
    subgraph_list.append(subgraph)

'''
Remove a node from big graph and enumerate isomorphims again
'''

node_to_remove = 'C'
big_graph.remove_node(node_to_remove)

GM2 = iso.GraphMatcher(big_graph, small_graph)
subgraph_list_2 = []
for subgraph in GM2.subgraph_isomorphisms_iter():
    subgraph_list_2.append(subgraph)

'''
Remove subgraphs isomorphisms which include the removed node
'''

subgraph_list = [subgraph for subgraph in subgraph_list if not node_to_remove in subgraph]

'''
Add a new node to big graph
'''

new_node = 'G'
big_graph.add_node(new_node)
big_graph.add_edges_from([['A', new_node], ['B', new_node]])

# count all isomorphisms
GM3 = iso.GraphMatcher(big_graph, small_graph)
subgraph_list_3 = []
for subgraph in GM3.subgraph_isomorphisms_iter():
    subgraph_list_3.append(subgraph)

# Take radius around new node
neighb = nx.ego_graph(big_graph, new_node, radius = 1)

GM4 = iso.GraphMatcher(neighb, small_graph)
for subgraph in GM4.subgraph_isomorphisms_iter():
    if new_node in subgraph:    
        subgraph_list.append(subgraph)
    


'''
Print out all isomorphism lists
'''

print '\nSubgraphs list after manipulation'
for subgraph in subgraph_list:
    print subgraph
    
print '\nSubgraphs enumerated after removing C'
for subgraph in subgraph_list_2:
    print subgraph
    
print '\nSubgraphs enumerated after removing C and adding G'
for subgraph in subgraph_list_3:
    print subgraph 

'''
Draw graphs
'''

#d = {}
#for i in range(len(self.KMC_lat.site_type_inds)):
#    d[i] = self.KMC_lat.cart_coords[i,:]
#nx.draw(big_graph)
#plt.draw()