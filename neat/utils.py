import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

# Logging or other utility functions can go here
def log_population_stats(population):
    # For example, you could compute average fitness
    avg_fitness = sum(g.fitness for g in population.genomes.values()) / len(population.genomes)
    print(f"Generation: {population.generation}, Avg Fitness: {avg_fitness}")
