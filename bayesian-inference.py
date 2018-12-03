import sys
import queue
from functools import reduce
from copy import copy, deepcopy
from itertools import combinations

NAME = 0
PARENTS = 1
VALUES = 2

class Node:
    def __init__(self, id, parents, values):
        self.id = id
        self.values = values
        self.parents = parents

        # No neighbours initially
        self.neighbours = []

    def __repr__(self):
        return self.id + ' ; ' + str(self.parents) + ' ; ' + str(self.values) + ' ; ' + str(self.neighbours) + '\n'

    def __eq__(self, other):
        if isinstance(other, Node):
            return self.id == other.id
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

class Graph:
    '''
    The nodes will be identified by id. The graph is a dict { node_id : node }
    Each node contains a list of its neighbours.
    '''
    def __init__(self, directed):
        self.nodes = {}
        self.directed = directed

    def __repr__(self):
        return str(list(self.nodes.values()))

    def get_size(self):
        return len(self.nodes)

    def get_node(self, id):
        return self.nodes[id]

    def get_neighbours(self, id):
        return [self.get_node(n_id) for n_id in self.nodes[id].neighbours]

    def add_node(self, node):
        self.nodes[node.id] = node

        # Connect to parents
        for p_id in node.parents:
            self.add_edge(p_id, node.id)

    def add_edge(self, u, v):
        self.nodes[u].neighbours.append(v)

        # Add the reverse if graph is not directed
        if not self.directed:
            self.nodes[v].neighbours.append(u)

    def remove_node(self, id):
        # Remove the node from neighbour lists
        for u in self.nodes.values():
            if id in u.neighbours:
                u.neighbours.remove(id)

        # Remove the node from the graph
        del self.nodes[id]

    def remove_edge(self, u, v):
        self.nodes[u].neighbours.remove(v)

        # Remove the reverse if graph is not directed
        if not self.directed:
            self.nodes[v].neighbours.remove(u)

    def is_edge(self, u, v):
        return v in self.nodes[u].neighbours

    def get_unconnected_neighbours(self, id):
        '''
        Returns a list of pairs of all the unconnected
        neighbours of the given node
        '''
        unconnected = []
        for (u, v) in combinations(self.nodes[id].neighbours, 2):
            if not self.is_edge(u, v):
                unconnected.append((u, v))
        return unconnected

def main():
    network = Graph(True)

    with open('bn1', 'r') as input:
        # Get problem parameters
        var_count, test_count = map(int, input.readline().split())

        # Build the bayesian network
        for i in range(var_count):
            var_line = input.readline()
            data = var_line.split(';')

            # Get values
            var_values = list(map(float, data[VALUES].split()))

            # Create node
            var = Node(data[NAME].strip(), data[PARENTS].split(), var_values)
            network.add_node(var)

        # Moralize graph
        moralized_graph = Graph(False)

        for node in network.nodes.values():
            moralized_graph.add_node(Node(node.id, node.parents, node.values))

            # Add link between parents of each node
            for (p1, p2) in combinations(node.parents, 2):
                moralized_graph.add_edge(p1, p2)

        cordial_graph = moralized_graph
        workspace = deepcopy(cordial_graph)

        while workspace.nodes:
            # Sort after the number of edges that will be added
            sorted_nodes = sorted(list(workspace.nodes.keys()), key=lambda u: len(workspace.get_unconnected_neighbours(u)))

            # Get the best node
            removed = sorted_nodes[0]
            
            # Add new edges between neighbours
            for (u, v) in workspace.get_unconnected_neighbours(removed):
                cordial_graph.add_edge(u, v)
                workspace.add_edge(u, v)

            # Remove selected node
            workspace.remove_node(removed)

        print(cordial_graph)

if __name__ == "__main__":
    main()