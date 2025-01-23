import numpy as np
import random
import copy
import pickle
import itertools
import math
import multiprocessing as mp

from neat.hyper_parameters import Hyperparameters
from neat import utils

def distance(a, b, distance_weights):
        """Calculate the genomic distance between two genomes."""
        a_edges = set(a._edges)
        b_edges = set(b._edges)

        # Does not distinguish between disjoint and excess
        matching_edges = a_edges & b_edges
        disjoint_edges = (a_edges - b_edges) | (b_edges - a_edges)
        N_edges = len(max(a_edges, b_edges, key=len))
        N_nodes = min(a._max_node, b._max_node)

        weight_diff = 0
        for i in matching_edges:
            weight_diff += abs(a._edges[i].weight - b._edges[i].weight)

        bias_diff = 0
        for i in range(N_nodes):
            bias_diff += abs(a._nodes[i].bias - b._nodes[i].bias)

        t1 = distance_weights['edge'] * len(disjoint_edges)/N_edges
        t2 = distance_weights['weight'] * weight_diff/len(matching_edges)
        t3 = distance_weights['bias'] * bias_diff/N_nodes
        return t1 + t2 + t3


def crossover(genome_a, genome_b):
        """Breed two genomes and return the child. Matching genes
        are inherited randomly, while disjoint genes are inherited
        from the fitter parent.
        """
        # Template genome for child
        child = Genome(genome_a._inputs, genome_a._outputs, genome_a._default_activation)
        a_in = set(genome_a._edges)
        b_in = set(genome_b._edges)

        # Inherit homologous gene from a random parent
        for i in a_in & b_in:
            parent = random.choice([genome_a, genome_b])
            child._edges[i] = copy.deepcopy(parent._edges[i])

        # Inherit disjoint/excess genes from fitter parent
        if genome_a._fitness > genome_b._fitness:
            for i in a_in - b_in:
                child._edges[i] = copy.deepcopy(genome_a._edges[i])
        else:
            for i in b_in - a_in:
                child._edges[i] = copy.deepcopy(genome_b._edges[i])
        
        # Calculate max node
        child._max_node = 0
        for (i, j) in child._edges:
            current_max = max(i, j)
            child._max_node = max(child._max_node, current_max)
        child._max_node += 1

        # Inherit nodes
        for n in range(child._max_node):
            inherit_from = []
            if n in genome_a._nodes:
                inherit_from.append(genome_a)
            if n in genome_b._nodes:
                inherit_from.append(genome_b)

            random.shuffle(inherit_from)
            parent = max(inherit_from, key=lambda p: p._fitness)
            child._nodes[n] = copy.deepcopy(parent._nodes[n])

        child.reset() 
        return child

class Edge(object):
    """A gene object representing an edge in the neural network."""
    def __init__(self, weight):
        self.weight = weight
        self.enabled = True


class Node(object):
    """A gene object representing a node in the neural network."""
    def __init__(self, activation):
        self.output = 0
        self.bias = 0
        self.activation = activation


class Genome:
    """
    Represents a single genome with node genes and connection genes.
    """
    def __init__(self, inputs, outputs, default_activation):
        # Nodes
        self._inputs = inputs
        self._outputs = outputs

        self._unhidden = inputs + outputs
        self._max_node = inputs + outputs

        # Structure
        self._edges = {} # (i, j) : Edge
        self._nodes = {} # NodeID : Node

        self._default_activation = default_activation

        # Performance
        self._fitness = 0
        self._adjusted_fitness = 0
        
    def generate_minimal(self):
        """
        Creates a new genome with a minimal structure:
        - Input nodes at IDs 0..(inputs-1)
        - Output nodes at IDs inputs..(inputs+outputs-1)
        - A fully connected set of edges from each input to each output
        """
        # 1) Create input nodes
        for i in range(self._inputs):
            self._nodes[i] = Node(self._default_activation)
        
        # 2) Create output nodes (right after the last input)
        for i in range(self._outputs):
            output_id = self._inputs + i
            self._nodes[output_id] = Node(self._default_activation)
        
        # 3) Update unhidden and max_node
        self._unhidden = self._inputs + self._outputs
        self._max_node = self._unhidden  # no hidden nodes yet
        
        # 4) Connect every input to every output
        for i in range(self._inputs):
            for j in range(self._inputs, self._unhidden):
                self._edges[(i, j)] = Edge(random.uniform(-1, 1))

    # def mutate(self):
    #     """
    #     Apply mutations (weight changes, structural changes, etc.).
    #     Fill out the details for:
    #       - Weight perturbation
    #       - Adding a connection
    #       - Adding a node
    #     """
    #     mutation_probabilities = Hyperparameters.mutation_probabilities
        
    #     for mutation, probability in mutation_probabilities.items():
    #         if random.random() < probability:
    #             match mutation:
    #                 case 'node': self.add_node()
    #                 case 'edge': self.add_connection()
    #                 case 'weight_perturb': self.shift_weight('weight_perturb')
    #                 case 'weight_set': self.shift_weight('weight_set')
    #                 case 'bias_perturb': self.shift_bias('bias_perturb')
    #                 case 'bias_set': self.shift_bias('bias_set')
    #                 case _: pass
    
    def mutate(self, probabilities):
        """Randomly mutate the genome to initiate variation."""
        if self.is_disabled():
            self.add_enabled()

        population = list(probabilities.keys())
        weights = [probabilities[k] for k in population]
        choice = random.choices(population, weights=weights)[0]

        if choice == "node":
            self.add_node()
        elif choice == "edge":
            (i, j) = self.random_pair()
            self.add_edge(i, j, random.uniform(-1, 1))
        elif choice == "weight_perturb" or choice == "weight_set":
            self.shift_weight(choice)
        elif choice == "bias_perturb" or choice == "bias_set":
            self.shift_bias(choice)

        self.reset()
                    

    def forward(self, inputs):
        """
        Compute output given some inputs using the genome's topology.
        You can do this by building a graph of nodes or topologically sorting
        the nodes and then computing outputs in order.
        """
        if len(inputs) != self._inputs:
            raise ValueError("Incorrect number of inputs.")

        # Set input values
        for i in range(self._inputs):
            self._nodes[i].output = inputs[i]
        
        # Generate backward-adjacency list 
        _from = {}
        for n in range(self._max_node):
            _from[n] = []

        for (i, j) in self._edges:
            if not self._edges[(i, j)].enabled:
                continue
            _from[j].append(i)

        # Calculate output values for each node
        ordered_nodes = itertools.chain(
            range(self._unhidden, self._max_node),
            range(self._inputs, self._unhidden)
        )
        for j in ordered_nodes:
            ax = 0
            for i in _from[j]:
                ax += self._edges[(i, j)].weight * self._nodes[i].output

            node = self._nodes[j]
            node.output = utils.sigmoid(ax + node.bias) # TODO activation function should be applied here (node.activation)
        
        return [self._nodes[n].output for n in range(self._inputs, self._unhidden)]
    
    
    
    # def add_edge(self):
    #     """Add a new connection gene to the genome."""
    #     # Find two unconnected nodes
    #     i = random.randint(0, self._max_node - 1)
    #     j = random.randint(self._inputs, self._max_node - 1)
    #     while (i, j) in self._edges:
    #         i = random.randint(0, self._max_node - 1)
    #         j = random.randint(self._inputs, self._max_node - 1)
        
    #     self._edges[(i, j)] = Edge(random.uniform(-1, 1))
    
    def add_edge(self, i, j, weight):
        """Add a new connection between existing nodes."""
        if (i, j) in self._edges:
            self._edges[(i, j)].enabled = True
        else:
            self._edges[(i, j)] = Edge(weight)
            
        
    def add_node(self):
        """Add a new node gene to the genome."""
        # Find a random connection to split
        edge = random.choice(list(self._edges.keys()))
        while not self._edges[edge].enabled:
            edge = random.choice(list(self._edges.keys()))
        
        # Disable old edge
        self._edges[edge].enabled = False
        
        # Add new node and edges
        new_node = self._max_node
        self._nodes[new_node] = Node(self._default_activation)
        self._max_node += 1
        
        self._edges[(edge[0], new_node)] = Edge(1)
        self._edges[(new_node, edge[1])] = Edge(self._edges[edge].weight)
        
    def add_enabled_connection(self):
        """Add a new connection gene to the genome."""
        # Find two unconnected nodes
        i = random.randint(0, self._max_node - 1)
        j = random.randint(self._inputs, self._max_node - 1)
        while (i, j) in self._edges:
            i = random.randint(0, self._max_node - 1)
            j = random.randint(self._inputs, self._max_node - 1)
        
        self._edges[(i, j)] = Edge(random.uniform(-1, 1))
        
    def add_enabled_node(self):
        """Add a new node gene to the genome."""
        # Find a random connection to split
        edge = random.choice(list(self._edges.keys()))
        while not self._edges[edge].enabled:
            edge = random.choice(list(self._edges.keys()))
        
        # Disable old edge
        self._edges[edge].enabled = False
        
        # Add new node and edges
        new_node = self._max_node
        self._nodes[new_node] = Node(self._default_activation)
        self._max_node += 1
        
        self._edges[(edge[0], new_node)] = Edge(1)
        self._edges[(new_node, edge[1])] = Edge(self._edges[edge].weight)
        
    def shift_weight(self, type):
        """Randomly shift, perturb, or set one of the edge weights."""
        e = random.choice(list(self._edges.keys()))
        if type == "weight_perturb":
            self._edges[e].weight += random.uniform(-1, 1)
        elif type == "weight_set":
            self._edges[e].weight = random.uniform(-1, 1)

    def shift_bias(self, type):
        """Randomly shift, perturb, or set the bias of an incoming edge."""
        # Select only nodes in the hidden and output layer
        n = random.choice(range(self._inputs, self._max_node))
        if type == "bias_perturb":
            self._nodes[n].bias += random.uniform(-1, 1)
        elif type == "bias_set":
            self._nodes[n].bias = random.uniform(-1, 1)
            
    def random_pair(self):
        """Generate random nodes (i, j) such that:
        1. i is not an output
        2. j is not an input
        3. i != j
        """
        i = random.choice([n for n in range(self._max_node) if not self.is_output(n)])
        j_list = [n for n in range(self._max_node) if not self.is_input(n) and n != i]

        if not j_list:
            j = self._max_node
            self.add_node()
        else:
            j = random.choice(j_list)

        return (i, j)
    
    def is_input(self, n):
        return n < self._inputs
    
    def is_output(self, n):
        return n >= self._unhidden
    
    def is_hidden(self, n):
        return self._inputs <= n < self._unhidden
    
    def is_enabled(self, i, j):
        return self._edges[(i, j)].enabled
    
    def is_disabled(self):
        """Determine if all of its genes are disabled."""
        return all(self._edges[i].enabled == False for i in self._edges)
    
    def disable(self, i, j):
        self._edges[(i, j)].enabled = False
        
    def enable(self, i, j):
        self._edges[(i, j)].enabled = True
        
    def get_fitness(self):
        return self._fitness
    
    def set_fitness(self, fitness):
        self._fitness = fitness
        
    def get_adjusted_fitness(self):
        return self._adjusted_fitness
    
    def set_adjusted_fitness(self, fitness):
        self._adjusted_fitness = fitness
        
    def get_nodes(self):
        return self._nodes
    
    def get_edges(self):
        return self._edges
    
    def get_max_node(self):
        return self._max_node
    
    def get_inputs(self):
        return self._inputs
    
    def reset(self):
        """Reset the state of the genome."""
        for n in self._nodes:
            self._nodes[n].output = 0
            
    def clone(self):
        """Create a deep copy of the genome."""
        return copy.deepcopy(self)
    

        
    def save(self, filename):
        """Save the genome to a file."""
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
            

    def distance(self, other_genome):
        """
        Calculate compatibility distance between this genome and another
        for speciation.
        """
        distance = 0.0
        # Implement distance calculation
        return distance
    
    
    

