import gymnasium as gym
from neat.config import NEATConfig
from neat.population import Population
from neat.utils import log_population_stats

def main():
    # Create environment. Use built-in Gym environment, e.g., CartPole-v1
    # or use your custom environment from gym_envs
    env = gym.make("CartPole-v1", render_mode=None)
    
    # Initialize config
    config = NEATConfig()
    # Adjust config.num_inputs / config.num_outputs to match env
    # For CartPole-v1, obs is shape (4,), action is discrete 2
    config.num_inputs = 4
    config.num_outputs = 2
    
    # Create population
    population = Population(config)
    
    max_generations = 50
    for gen in range(max_generations):
        population.run_generation(env)
        log_population_stats(population)
        
        # Check if a stopping condition is met, e.g., max fitness
        best_fitness = max(g.fitness for g in population.genomes.values())
        if best_fitness >= 200:
            print(f"Solved in generation {gen}!")
            break

    env.close()

if __name__ == "__main__":
    main()
