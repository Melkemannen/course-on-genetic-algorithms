

## Course Outline: Introduction to NEAT and Genetic Algorithms

## 1. Welcome and Course Overview (5 minutes)

### Objectives of the course
-[] Understand the basics of genetic algorithms and NEAT.
-[] Implement and experiment with NEAT in Jupyter Notebook.
-[] Structure: Presentation + hands-on coding.

## 2. Introduction to Genetic Algorithms (10 minutes)

### Presentation Topics:

### What is a Genetic Algorithm?
        Definition and analogy to natural selection.
        Key concepts: population, fitness, selection, crossover, mutation.
### Why Use Genetic Algorithms?
        Advantages: heuristic search, exploration of complex solution spaces.
        Applications: optimization, scheduling, game AI, neural network training.
### Applications Overview:
        Examples: evolving images, DALL-E style creativity (brief touch), game strategies.
        Showcases: GIFs of CrawlAI, NEAT Tactics.

### Activity (5 min):
        Discuss in groups where genetic algorithms can be useful


## 3. Introduction to NEAT (20 minutes)

## Presentation Topics:

-[] What is NEAT?
        Basics: evolving neural networks.
        Comparison to standard genetic algorithms.
        Benefits of NEAT: preserving innovation, speciation.

-[] Key Concepts of NEAT:
        - Genome representation: nodes, connections, weights.
        - Input and output in environment
        - Fitness
        - Mutation types: weight mutation, adding nodes, adding connections.
        - Speciation: maintaining diversity in the population.
        - Crossover


### Activity/ short break
Discuss the key concepts in Neat in pairs.
- How would you calculate a fitness function?
- Why do we use speciation?
- What are the different types of mutations?
- How do you do crossover?

## Start NEAT implementation details
- Environment
- Find the inputs and outputs
- Determining the structure of the neural network
- Initialize population
- Fitness

Population Initialization:

    Representation of individuals (neural network genomes).
    Initial parameters: number of inputs, outputs, population size.

Fitness Function:

    Role of fitness in evolution.
    Designing effective fitness functions for NEAT (e.g., reward specific behaviors).
    Examples: solving a maze, balancing a pole, playing a game.

## CODE 

## Mutation
- implementation details

## CODE Mutation (10 min)

## Speciation and Crossover
Speciation in NEAT:

    Preventing premature convergence.
    Speciation mechanics: sharing, distance calculation.

Selection and Crossover:

    How NEAT combines genomes while preserving innovation.
    Key challenges: matching genes, handling compatibility.

## CODE Speciation and Crossover, put it all together and train your agents. 

## Experimentation:
    Test NEAT on another problem (e.g., Flappy Bird simulation or a simple game environment like OpenAI Gym).

## 7. Wrap-Up and Q&A (10-15 minutes)

### Recap of key concepts:
        Genetic Algorithms: population, fitness, selection, crossover, mutation.
        NEAT: genome evolution, speciation, and innovation.

## Anounce winner of competition