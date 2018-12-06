import sys
from functools import reduce
from copy import copy, deepcopy
import queue
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

        # Messages from children
        self.received = {}

    def __repr__(self):
        return str(self.vars)

def multiply(phi1, phi2):
    '''
    Multiplies 2 factors and returns the result as a new factor.
    '''
    # Sanity checks
    if phi1 is None:
        return deepcopy(phi2)

    if phi2 is None:
        return deepcopy(phi1)

    result = Factor([], {})

    # Resulting factor joins the vars in phi1 and phi2
    result.vars = list(set(phi1.vars) | set(phi2.vars))

    # Find variables on which to join
    common_vars = list(set(phi1.vars) & set(phi2.vars))

    for binding1, val1 in phi1.values.items():
        for binding2, val2 in phi2.values.items():
            match = True

            # Check to see if the bindings match on
            # the common variables
            for var in common_vars:
                index1 = phi1.vars.index(var)
                index2 = phi2.vars.index(var)

                if binding1[index1] != binding2[index2]:
                    match = False

            if match:
                binding = [0 for x in result.vars]

                # Add bindings of vars from the first factor
                for i in range(len(binding1)):
                    index = result.vars.index(phi1.vars[i])
                    binding[index] = binding1[i]

                # Add bindings of vars from the second factor
                for i in range(len(binding2)):
                    if phi2.vars[i] not in common_vars:
                        index = result.vars.index(phi2.vars[i])
                        binding[index] = binding2[i]

                # Multiply values
                result.values[tuple(binding)] = val1 * val2

    return result

def sum_out(var, phi):
    '''
    Removes the var from the factor
    by summin over its values.
    Returns the result as a new factor
    '''
    # Sanity checks
    if not phi:
        return

    if var not in phi.vars:
        return deepcopy(phi)

    result = Factor(deepcopy(phi.vars), {})

    # Remove the var from the new factor
    result.vars.remove(var)
    index = phi.vars.index(var)

    for (binding, val) in phi.values.items():
        new_binding = ()

        # Construct new binding wthout the removed var
        for i in range(len(binding)):
            if i != index:
                new_binding = new_binding + (binding[i],)

        # Add the new binding to the resulting factor
        if new_binding in result.values:
            result.values[new_binding] += val
        else:
            result.values[new_binding] = val

    return result

def condition_factor(phi, Z):
    '''
    Returns a new factor the contains
    only the lines that correspond to the bindings in Z
    '''
    if not phi:
        return

    result = deepcopy(phi)

    to_remove = []
    for (binding, val) in result.values.items():
        for var in Z.keys():
            if var not in phi.vars:
                continue

            # Check if binding matches with Z
            index = phi.vars.index(var)
            if binding[index] != Z[var]:
                to_remove.append(binding)
                break

    # Remove extra bindings
    for b in to_remove:
        del result.values[b]

    return result

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

def traverse_tree(mst, root, Z):
    q = queue.Queue()
    visited = {node : False for node in mst.nodes.keys()}

    q.put(root)
    visited[root] = True

    while not q.empty():
        u = q.get()
        mst.nodes[u].factor = condition_factor(mst.nodes[u].factor, Z)

        for v in mst.neighbours[u]:
            if not visited[v]:
                visited[v] = True
                mst.nodes[u].children.add(mst.nodes[v])
                mst.nodes[v].parent = mst.nodes[u]
                q.put(v)

def bottom_up_propagation(clique):
    # Update belief with the message from each child
    for c in clique.children:
        r = bottom_up_propagation(c)
        
        # Store the received message
        clique.received[c] = r
        
        # Update the belief
        clique.factor = multiply(clique.factor, r)

    # Root doesn't need to return a value
    if not clique.parent:
        return

    m = clique.factor
    
    # Project message on common vars 
    out_vars = clique.vars  - (clique.vars & clique.parent.vars)
    for var in out_vars:
        m = sum_out(var, m)

    return m

def top_down_propagation(clique, m):
    # Update belief
    clique.factor = multiply(clique.factor, m)

    for c in clique.children:
        # Compute message
        prev = clique.received[c]
        for (b, val) in prev.values.items():
            prev.values[b] = 1 / val

        m = multiply(clique.factor, prev)

        # Project message on coomn vars
        out_vars = clique.vars  - (clique.vars & c.vars)
        for var in out_vars:
            m = sum_out(var, m)

        # Propagate message to child
        top_down_propagation(c, m)

def belief_propagation(root):
    '''
    Propagate beliefs across the clique tree
    '''
    # Propagate from leaves to root
    bottom_up_propagation(root)
    
    # Propagate from root to leaves
    top_down_propagation(root, None)

def get_dictionary_from_input(input):
    # Remove ' '
    vars = input.split()

    d = {}
    for var in vars:
        # Separate variable name and value
        k, v = var.split('=')
        d[k] = int(v)
    return d

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
        for var in network.nodes.values():
            factor = Factor.from_variable(var)

            # Assign factor to a clique
            for (vars, clique) in mst.nodes.items():
                if not set(factor.vars) - vars:
                    clique.factor = multiply(clique.factor, factor)
                    break

        # Choose a 'random' node as root
        root = list(mst.nodes.keys())[0]

        # Run tests
        results = []
        for i in range(test_count):
            test_line = input.readline()

            # Separate inference from observations
            data = test_line.split('|')

            # Obtain dictionary of observations
            observations = get_dictionary_from_input(data[1])

            # Obtain dictionary of inferences
            infs = get_dictionary_from_input(data[0])

            test_mst = deepcopy(mst)

            # Condition tree based on observations
            traverse_tree(test_mst, root, observations)

            # Belief propagation
            belief_propagation(test_mst.nodes[root])

            required_vars = set(infs.keys())

            available = list(filter(lambda clique: not (set(required_vars) - set(clique.factor.vars)), test_mst.nodes.values()))

            # Skip bonus tests
            if not available:
                results.append(0.0)
                continue
            
            # Choose a factor that contains all the required vars
            good_factor = available[0].factor

            out_vars = set(good_factor.vars) - required_vars

            for var in out_vars:
                good_factor = sum_out(var, good_factor)

            inferred_tuple = [0 for var in required_vars]
            for (var, val) in infs.items():
                inferred_tuple[good_factor.vars.index(var)] = val

            p = good_factor.values[tuple(inferred_tuple)] / sum(good_factor.values.values())
            results.append(p)

        # Verify results
        count = 0
        for i in range(test_count):
            test_result = float(input.readline())

            # Check for floating point errors
            if abs(results[i] - test_result) < 0.001:
                print("Test #{} PASSED".format(i))
                count += 1
            else:
                print("Test #{} FAILED: Expected {}, Got {}".format(i, test_result, results[i]))

        # Print final score
        print("Total: {}/{}".format(count, test_count))

if __name__ == "__main__":
    main()