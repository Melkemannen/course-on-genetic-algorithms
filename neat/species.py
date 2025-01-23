import math
import random

from neat.genome import crossover
from neat.genome import distance
from neat.genome import Genome


class Specie:
    """
    Represents a species in NEAT. Maintains a list of genomes that are
    compatible with its representative.
    """
    def __init__(self, max_fitness_history, *members):
        self._members = list(members)
        self._fitness_history = []
        self._fitness_sum = 0
        self._max_fitness_history = max_fitness_history
        
    def breed(self, mutation_probabilities, breed_probabilities):
        """Return a child as a result of either a mutated clone
        or crossover between two parent genomes.
        """
        # Either mutate a clone or breed two random genomes
        population = list(breed_probabilities.keys())
        probabilities= [breed_probabilities[k] for k in population]
        choice = random.choices(population, weights=probabilities)[0]

        if choice == "asexual" or len(self._members) == 1:
            child: Genome = random.choice(self._members).clone()
            child.mutate(mutation_probabilities)
        elif choice == "sexual":
            (mom, dad) = random.sample(self._members, 2)
            child = crossover(mom, dad)

        return child
        
    def add_member(self, genome: Genome):
        self._members.append(genome)
        if genome._fitness > self.best_fitness:
            self.best_fitness = genome._fitness
    
    def update_fitness(self):
        """Update the adjusted fitness values of each genome 
        and the historical fitness."""
        for genome in self._members:
            genome._adjusted_fitness = genome._fitness/len(self._members)

        self._fitness_sum = sum([genome._adjusted_fitness for genome in self._members])
        self._fitness_history.append(self._fitness_sum)
        if len(self._fitness_history) > self._max_fitness_history:
            self._fitness_history.pop(0)
    
    
    def cull_genomes(self, fittest_only: bool = False):
        """Exterminate the weakest genomes per specie."""
        self._members.sort(key=lambda genome: genome._fitness, reverse=True)
        if fittest_only:
            # Only keep the winning genome
            remaining = 1
        else:
            # Keep top 25%
            remaining = int(math.ceil(0.25*len(self._members)))

        self._members = self._members[:remaining]

    def get_best(self):
        """Get the member with the highest fitness score."""
        return max(self._members, key=lambda g: g._fitness)
    
    def get_average_fitness(self):
        """Get the average fitness of the species."""
        return self._fitness_sum/len(self._members)
    
    def can_progress(self):
        """Determine whether species should survive the culling."""
        n = len(self._fitness_history)
        avg = sum(self._fitness_history) / n
        return avg > self._fitness_history[0] or n < self._max_fitness_history


