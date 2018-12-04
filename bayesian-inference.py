import sys
from functools import reduce
from copy import copy, deepcopy
from itertools import combinations, product
from graph import *

DOMAIN = [0, 1]

NAME = 0
PARENTS = 1
VALUES = 2

class Variable:
    def __init__(self, id, parents, values):
        self.id = id
        self.values = values
        self.parents = parents

    def __repr__(self):
        return self.id + ' ; ' + self.parents + ' ; ' + self.values

class Factor:
    def __init__(self, vars, values):
        self.vars = vars
        self.values = values

    @classmethod
    def from_variable(cls, var):
        '''
        Generates a new factor from an existing variable
        '''
        factor = cls([], {})

        # Add the variables to the factor
        factor.vars = [var.id] + var.parents

        # Generate all possible bindings of the variables
        for binding in product(DOMAIN, repeat=len(factor.vars)):
            # Compute the index of the binding
            # to get the corresponding probability
            index = 0
            for i in range(1, len(binding)):
                if binding[i] == 1:
                    index += 1 << (len(binding) - 1 - i)

            # Check if the variable is negated
            if binding[0] == 0:
                factor.values[binding] = 1 - var.values[index]
            else:
                factor.values[binding] = var.values[index]

        return factor

    def __repr__(self):
        return str(self.vars) + ' ; ' + str(self.values)

    def nice_print(self, indent="\t"):
        '''
        Print factor in an easy to read format
        '''
        line = " | ".join(self.vars + ["ϕ(" + ",".join(self.vars) + ")"])
        sep = "".join(["+" if c == "|" else "-" for c in list(line)])

        print(indent + sep)
        print(indent + line)
        print(indent +sep)

        for values, p in self.values.items():
            print(indent + " | ".join([str(v) for v in values] + [f"{p:.6f}"]))

        print(indent + sep)

class CliqueNode:
    '''
    A node in the clique graph. Uses the variables in the clique
    as an id. Has a factor associated with it.
    '''
    def __init__(self, vars):
        self.vars = vars

        # Factor associated with the clique
        self.factor = None

        # Traversal params
        self.parent = None
        self.children = set()

    def __repr__(self):
        return str(self.vars)

def make_clique_graph(cliques):
    '''
    Creates a graph of CliqueNodes from a list of
    cliques, where each clique is a set of node ids.
    '''
    graph = Graph(False)

    # Generate clique nodes from lists of variables
    cliques = [CliqueNode(frozenset(c)) for c in cliques]

    # Add edge between 2 cliques if they intersect
    for c1 in cliques:
        for c2 in cliques:
            if c1 != c2 and c1.vars & c2.vars:
                # Add clique nodes
                graph.add_node(c1.vars, c1)
                graph.add_node(c2.vars, c2)
                
                # Add edge
                graph.add_edge(c1.vars, c2.vars)

    return graph

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
            var = Variable(data[NAME].strip(), data[PARENTS].split(), var_values)
            network.add_node(var.id, var)

            # Add edges from parents
            for p_id in var.parents:
                network.add_edge(p_id, var.id)

        # Moralize graph
        work_graph = moralize(network)

        # Cordial graph
        work_graph = chordal(work_graph)

        # Get list of cliques
        cliques = work_graph.get_maximal_cliques(set(), set(work_graph.nodes.keys()), set())

        # Build graph of cliques
        clique_graph = make_clique_graph(cliques)

        # Get the MaxST
        mst = clique_graph.get_mst(lambda edge: len(edge[0] & edge[1]), True)

        # Generate initial factors
        # for var in network.nodes.values():
            # Factor.from_variable(var).nice_print()

if __name__ == "__main__":
    main()