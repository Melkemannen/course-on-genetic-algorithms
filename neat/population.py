import random
from neat.species import Specie
from neat.genome import Genome

class Population:
    def __init__(self, config):
        self.config = config
        self.generation = 0
        
        # Dictionary: genome_key -> Genome
        self.genomes = {}
        
        # List of Species
        self.species_list = []
        
        self.initialize_population()

    def initialize_population(self):
        """
        Creates the initial population of genomes.
        """
        for i in range(self.config.population_size):
            genome = Genome(self.config, key=i)
            self.genomes[i] = genome

    def speciate(self):
        """
        Assign each genome to a species based on compatibility distance.
        """
        # Clear out old species members
        for species in self.species_list:
            species.members.clear()

        # For each genome, find a species or create a new one
        for genome in self.genomes.values():
            placed = False
            for species in self.species_list:
                distance = genome.distance(species.representative)
                if distance < self.config.compatibility_threshold:
                    species.add_member(genome)
                    placed = True
                    break
            if not placed:
                new_species = Specie(genome)
                new_species.add_member(genome)
                self.species_list.append(new_species)

    def reproduce(self):
        """
        Produce the next generation of genomes through sexual / asexual reproduction.
        """
        next_gen = {}
        # Example approach: keep track of total fitness, pick parents, do crossover...
        # For now, let's just do random selection and no crossover for demonstration.
        
        # Flatten all members
        all_members = []
        for species in self.species_list:
            all_members.extend(species.members)

        # Sort by fitness descending
        all_members.sort(key=lambda x: x.fitness, reverse=True)

        # Simple elitism or random picks
        for i in range(self.config.population_size):
            parent = random.choice(all_members)
            child = Genome(self.config, key=i)
            # Possibly do crossover with second parent
            # Possibly mutate
            child.mutate()
            next_gen[i] = child
        
        self.genomes = next_gen
        self.generation += 1

    def evaluate_population(self, env):
        """
        Evaluate the fitness of each genome in the given environment.
        """
        for genome in self.genomes.values():
            obs, _ = env.reset()
            done = False
            total_reward = 0.0
            while not done:
                # forward pass
                action = genome.feed_forward(obs)
                # If discrete action, pick argmax
                action = int(action.argmax()) 
                
                obs, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                
                done = terminated or truncated
            
            genome.fitness = total_reward

    def run_generation(self, env):
        # Evaluate
        self.evaluate_population(env)
        # Speciate
        self.speciate()
        # Reproduce
        self.reproduce()


