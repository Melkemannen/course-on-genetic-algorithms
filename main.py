import gymnasium as gym
import numpy as np

from neat.hyper_parameters import Hyperparameters
from neat.genetic_algorithm import GeneticAlgorithm

def run_episode(env, genome):
    """
    Runs one episode of CartPole-v1 using the given genome to choose actions.
    Returns the total reward (fitness) of this episode.
    """
    obs, info = env.reset()
    done = False
    total_reward = 0.0
    
    while not done:
        # Convert the observation to a Python list (if needed)
        obs_list = obs.tolist() if hasattr(obs, "tolist") else obs
        
        # Forward pass through the genome: get action values
        action_values = genome.forward(obs_list)
        
        # Assuming discrete action space, pick the argmax
        # (For multi-discrete or continuous, adjust accordingly)
        #action = np.argmax(action_values)
        
        action = max(0, min(1, int(np.argmax(action_values))))
        
        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        total_reward += reward
    
    return total_reward

def main():
    # Create the Gymnasium environment (CartPole-v1)
    env = gym.make("CartPole-v1", render_mode=None)

    # Set hyperparameters (tweak these as needed)
    hp = Hyperparameters()
    hp.max_generations = 50  # Maximum generations to evolve
    # Optional: adjust additional hyperparams, e.g.:
    # hp.distance_weights["bias"] = 0.4
    # hp.mutation_probabilities["weight_perturb"] = 0.3
    
    # For CartPole: 4 inputs (state space), 2 outputs (left or right)
    brain = GeneticAlgorithm(inputs=4, outputs=4, population=100, hyperparams=hp)
    brain.generate()  # Create the initial population of genomes

    generation_count = 0
    while brain.should_evolve():
        # Get the current genome from the population
        genome = brain.get_current()

        # Evaluate it in an episode and set its fitness
        fitness_score = run_episode(env, genome)
        genome.set_fitness(fitness_score)

        # Move on to the next genome
        brain.next_iteration()
        
        # Check if we just finished an entire generation
        # (Every population=100 genomes => 1 generation)
        if brain._current_genome == 0:
            generation_count += 1
            best_genome = brain.get_fittest()
            print(f"Generation {generation_count} | Best Fitness So Far: {best_genome._fitness:.2f}")

            # (Optional) early stopping if you exceed a threshold
            if best_genome._fitness >= 200:
                print(f"Solved in generation {generation_count}!")
                break
    
    # Grab the best genome from the final run
    best_genome = brain.get_fittest()
    print(f"Final Best Fitness: {best_genome._fitness:.2f}")
    
    # Save the brain to disk (if desired)
    # brain.save('cartpole_brain.neat')

    env.close()

if __name__ == "__main__":
    main()
