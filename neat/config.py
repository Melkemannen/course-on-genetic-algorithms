class NEATConfig:
    """
    Holds NEAT configuration parameters.
    Customize these for your specific use-case or environment.
    """
    def __init__(self):
        # Population size
        self.population_size = 150
        
        # Speciation settings
        self.compatibility_threshold = 3.0
        
        # Mutation rates
        self.weight_mutation_rate = 0.8
        self.weight_perturbation_rate = 0.9
        self.new_node_rate = 0.03
        self.new_connection_rate = 0.05
        
        # Activation function
        self.activation = 'sigmoid'
        
        # Number of inputs and outputs (set these dynamically later if needed)
        self.num_inputs = 4
        self.num_outputs = 2
        
        # Other hyperparameters as needed...
