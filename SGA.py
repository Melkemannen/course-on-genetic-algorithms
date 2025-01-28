import numpy as np
import pandas as pd

# Load knapsack dataset
def load_knapsack_data():
    df = pd.read_csv("knapPI_12_500_1000_82.csv")
    return df.iloc[:, 1].values, df.iloc[:, 2].values  # Extract profits and weights

# Initialize population
def initialize_population(pop_size, chrom_length):
    return np.random.randint(2, size=(pop_size, chrom_length))

# Fitness Function
def fitness_function(chromosome, profits, weights, capacity, penalty_factor=50):
    total_profit = np.sum(profits[chromosome == 1])
    total_weight = np.sum(weights[chromosome == 1])
    
    if total_weight > capacity:
        penalty = penalty_factor * (total_weight - capacity)
        return total_profit - penalty  # Penalized fitness
    return total_profit  # Valid solution

# Crossover (Single-point crossover)
def crossover_single_point(parent1, parent2, crossover_rate=0.8):
    if np.random.rand() < crossover_rate:
        point = np.random.randint(1, len(parent1))
        child1 = np.concatenate((parent1[:point], parent2[point:]))
        child2 = np.concatenate((parent2[:point], parent1[point:]))
        return child1, child2
    return parent1, parent2

# Crossover (N-point crossover)
def crossover_n_points(parent1, parent2, n, crossover_rate=0.8):
    if np.random.rand() < crossover_rate:
        points = sorted(np.random.choice(range(1, len(parent1)), n, replace=False))
        child1, child2 = np.copy(parent1), np.copy(parent2)
        for i in range(0, len(points), 2):
            child1[points[i]:points[i+1]] = parent2[points[i]:points[i+1]]
            child2[points[i]:points[i+1]] = parent1[points[i]:points[i+1]]
        return child1, child2
    return parent1, parent2

# Mutation (Adaptive Mutation)
def adaptive_mutate(chromosome, generation, max_generations, mutation_rate=0.05):
    adjusted_mutation_rate = mutation_rate * (1 - (generation / max_generations))  # Reduce mutation over time
    mutation_mask = np.random.rand(len(chromosome)) < adjusted_mutation_rate
    chromosome[mutation_mask] = 1 - chromosome[mutation_mask]
    return chromosome

# Tournament Selection
def tournament_selection(population, fitnesses, tournament_size=3):
    tournament_indices = np.random.choice(len(population), tournament_size, replace=False)
    best_index = tournament_indices[np.argmax(fitnesses[tournament_indices])]
    return population[best_index]

# Select parents
def select_parents(population, fitnesses, tournament_size=3):
    return tournament_selection(population, fitnesses, tournament_size), \
           tournament_selection(population, fitnesses, tournament_size)

# Deterministic Crowding
def deterministic_crowding(parent1, parent2, child1, child2, fitness_function, profits, weights, capacity):
    fitness_p1 = fitness_function(parent1, profits, weights, capacity)
    fitness_p2 = fitness_function(parent2, profits, weights, capacity)
    fitness_c1 = fitness_function(child1, profits, weights, capacity)
    fitness_c2 = fitness_function(child2, profits, weights, capacity)
    
    new1 = child1 if fitness_c1 > fitness_p1 else parent1
    new2 = child2 if fitness_c2 > fitness_p2 else parent2
    return new1, new2

# Replacement (Elitism)
def replacement(population, new_population, profits, weights, capacity):
    all_solutions = np.vstack((population, new_population))
    fitnesses = np.array([fitness_function(ind, profits, weights, capacity) for ind in all_solutions])
    sorted_indices = np.argsort(fitnesses)[::-1]  # Sort in descending order
    return all_solutions[sorted_indices[:len(population)]]

# Main GA loop
def genetic_algorithm(pop_size, chrom_length, generations, profits, weights, capacity, mutation_rate, crossover_rate):
    population = initialize_population(pop_size, chrom_length)
    
    for gen in range(generations):
        fitnesses = np.array([fitness_function(chrom, profits, weights, capacity) for chrom in population])
        new_population = []
        
        while len(new_population) < pop_size:
            parent1, parent2 = select_parents(population, fitnesses)
            child1, child2 = crossover_n_points(parent1, parent2, 4, crossover_rate)
            child1, child2 = deterministic_crowding(parent1, parent2, child1, child2, fitness_function, profits, weights, capacity)
            new_population.append(adaptive_mutate(child1, gen, generations, mutation_rate))
            new_population.append(adaptive_mutate(child2, gen, generations, mutation_rate))
        
        population = replacement(population, np.array(new_population[:pop_size]), profits, weights, capacity)
        best_fitness = np.max([fitness_function(c, profits, weights, capacity) for c in population])
        print(f"Generation {gen + 1}: Best Fitness = {best_fitness}")

# Load data and run GA
profits, weights = load_knapsack_data()
capacity = 280785  
mutation_rates = [0.01, 0.05, 0.1]
crossover_rates = [0.7, 0.8, 0.9]
population_sizes = [50, 100, 200]
p = 50
m = 0.001
c = 0.8

print(f"Running GA with mutation={m}, crossover={c}, population={p}")
genetic_algorithm(p, len(profits), 50, profits, weights, capacity, m, c)
