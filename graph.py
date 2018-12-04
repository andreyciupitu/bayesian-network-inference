import sys
from copy import copy, deepcopy
from itertools import combinations

class Graph:
    '''
    The nodes will be identified by id. The graph is a dict
    { node_id : set(neighbour_id) }
    '''
    def __init__(self, directed):
        self.nodes = {}
        self.neighbours = {}
        self.parents = {}
        self.directed = directed

    def __repr__(self):
        return str(self.neighbours)

    def is_edge(self, u, v):
        return v in self.neighbours[u]

    def add_node(self, id, value=None):
        # Check if node already exists
        if id in self.nodes:
            return

        self.nodes[id] = value
        self.neighbours[id] = set()
        self.parents[id] = set()

    def add_edge(self, u, v):
        # Add nodes if they dont exist
        if u not in self.nodes:
            self.add_node(u)
        if v not in self.nodes:
            self.add_node(v)

        self.neighbours[u].add(v)
        self.parents[v].add(u)

        # Add the reverse if graph is not directed
        if not self.directed:
            self.neighbours[v].add(u)
            self.parents[u].add(v)

    def remove_node(self, id):
        # Check if node is in graph
        if id not in self.nodes:
            return

        # Remove edges connected to the node
        to_remove = [(parent, id) for parent in self.parents[id]]
        to_remove += [(id, nb) for nb in self.neighbours[id]]
        for (u, v) in to_remove:
            self.remove_edge(u, v)

        # Remove the node from the graph
        del self.nodes[id]
        del self.neighbours[id]
        del self.parents[id]

    def remove_edge(self, u, v):
        # Check if nodes are in graph
        if u not in self.nodes or v not in self.nodes:
            return

        if v in self.neighbours[u]:
            self.neighbours[u].remove(v)
            self.parents[v].remove(u)

        # Remove the reverse if graph is not directed
        if not self.directed:
            if u in self.neighbours[v]:
                self.neighbours[v].remove(u)
                self.parents[u].remove(v)

    def get_unconnected_neighbours(self, id):
        '''
        Returns a list of pairs of all the unconnected
        neighbours of the given node
        '''
        unconnected = []
        for (u, v) in combinations(self.neighbours[id], 2):
            if not self.is_edge(u, v):
                unconnected.append((u, v))
        return unconnected

    def get_maximal_cliques(self, R, P, X):
        '''
        Applies Bron-Kerbosh and returns a list of
        cliques. A clique is represented as a set of node ids.
        '''
        cliques = []

        if not P and not X:
            return [R]

        for v in P:
            set_v = set([v])
            neighbours = self.neighbours[v]

            # Find all clique extensions of R that contain v
            cliques += self.get_maximal_cliques(R | set_v, P & neighbours, X & neighbours)

            P = P - set_v
            X = X | set_v

        return cliques

    def get_mst(self, weight, reversed):
        '''
        Applies Kruskall do determine the MST, using the
        given weight function to sort the edges
        '''
        sets = {}

        # Init trees
        for u in self.nodes:
            sets[u] = set([u])

        # Get all edges
        edges = []
        for u in self.nodes.keys():
            for v in self.neighbours[u]:
                if self.directed or (v, u) not in edges:
                    edges.append((u, v))

        # Sort edges by weigths
        edges.sort(reverse=reversed, key=weight)

        # Determine MST
        mst = Graph(False)
        for (u, v) in edges:
            if sets[u] != sets[v]:
                # Add nodes to spanning tree
                mst.add_node(u, self.nodes[u])
                mst.add_node(v, self.nodes[v])
                mst.add_edge(u, v)

                # Join partial trees
                sets[u] = sets[u] | sets[v]
                sets[v] = sets[u]

                # Update all other trees
                for x in sets[u]:
                    sets[x] = sets[u]

        return mst

def moralize(g):
    '''
    Transforms the graph into a moral graph and returns the new graph
    '''
    # Graph must be directed
    if not g.directed:
        return None

    # Make undirected graph
    moralized_graph = Graph(False)

    # Transform directed graph into undirected graph
    for (id, node) in g.nodes.items():
        moralized_graph.add_node(id, node)

        # Add edges from parents
        for p_id in g.parents[id]:
            moralized_graph.add_edge(p_id, id)

        # Add link between parents of each node
        for (p1, p2) in combinations(g.parents[id], 2):
            moralized_graph.add_edge(p1, p2)

    return moralized_graph

def chordal(g):
    '''
    Transforms the graph into a chordal graph and returns the new graph
    '''
    chordal = deepcopy(g)
    tmp = deepcopy(g)
    while tmp.nodes:
        # Sort after the number of edges that will be added
        sorted_nodes = sorted(tmp.nodes.keys(), key=lambda u: len(tmp.get_unconnected_neighbours(u)))

        # Get the best node
        removed = sorted_nodes[0]

        # Add new edges between neighbours
        for (u, v) in tmp.get_unconnected_neighbours(removed):
            chordal.add_edge(u, v)
            tmp.add_edge(u, v)

        # Remove selected node
        tmp.remove_node(removed)

    return chordal