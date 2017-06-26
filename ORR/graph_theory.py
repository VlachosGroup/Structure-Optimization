# -*- coding: utf-8 -*-
"""
Created on Fri May 26 15:36:16 2017

@author: mpnun
"""

class Graph(object):

    '''
    Adapted from http://www.python-course.eu/graphs_python.php
    '''

    def __init__(self, graph_dict=None):
        """ initializes a graph object 
            If no dictionary or None is given, 
            an empty dictionary will be used
        """
        if graph_dict == None:
            graph_dict = {}
        self.__graph_dict = graph_dict

    def vertices(self):
        """ returns the vertices of a graph """
        return list(self.__graph_dict.keys())

    def edges(self):
        """ returns the edges of a graph """
        return self.__generate_edges()

    def add_vertex(self, vertex):
        """ If the vertex "vertex" is not in 
            self.__graph_dict, a key "vertex" with an empty
            list as a value is added to the dictionary. 
            Otherwise nothing has to be done. 
        """
        if vertex not in self.__graph_dict:
            self.__graph_dict[vertex] = []


    def remove_vertex(self, vertex):
        
        '''
        Delete a node from the graph
        '''        
        
        # Delete vertex from the neighbor list of all of its neighbors
        adj_vertices =  self.__graph_dict[vertex]
        for nn in adj_vertices:
            self.__graph_dict[nn].remove(vertex)
            
        # Remove vertex from the graph completely
        self.__graph_dict.pop(vertex, None)

    def add_edge(self, edge):
        """ assumes that edge is of type set, tuple or list; 
            between two vertices can be multiple edges! 
        """
        edge = set(edge)
        (vertex1, vertex2) = tuple(edge)
        if vertex1 in self.__graph_dict:
            self.__graph_dict[vertex1].append(vertex2)
        else:
            self.__graph_dict[vertex1] = [vertex2]
        if vertex2 in self.__graph_dict:
            self.__graph_dict[vertex2].append(vertex1)
        else:
            self.__graph_dict[vertex2] = [vertex1]


    def is_node(self,vertex):
        return vertex in self.__graph_dict

    def get_neighbors(self, vertex):
        return self.__graph_dict[vertex]

    def __generate_edges(self):
        """ A static method generating the edges of the 
            graph "graph". Edges are represented as sets 
            with one (a loop back to the vertex) or two 
            vertices 
        """
        edges = []
        for vertex in self.__graph_dict:
            for neighbour in self.__graph_dict[vertex]:
                if {neighbour, vertex} not in edges:
                    edges.append({vertex, neighbour})
        return edges

    def __str__(self):
        res = "vertices: "
        for k in self.__graph_dict:
            res += str(k) + " "
        res += "\nedges: "
        for edge in self.__generate_edges():
            res += str(edge) + " "
        return res
        
    def get_coordination_number(self, vertex):
        """ The degree of a vertex is the number of edges connecting
            it, i.e. the number of adjacent vertices. Loops are counted 
            double, i.e. every occurence of vertex in the list 
            of adjacent vertices. """ 
        return len(self.__graph_dict[vertex])

        
    def get_generalized_coordination_number(self, vertex, CN_max):
        
        """
        Compute the GCN of a vertex        
        """ 
        
        GCN = 0
        adj_vertices =  self.__graph_dict[vertex]
        for nn in adj_vertices:
            GCN += len( self.__graph_dict[nn] ) / float(CN_max)
        return GCN