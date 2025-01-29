
import flappy_bird_gym
import numpy as np
import time


import random
from SGANN_flappy.ipynb import construct_nn, forward_pass 


env = flappy_bird_gym.make("FlappyBird-v0")


# ===============================================================
# Neural network parameters:
# ===============================================================
num_inputs = 2 # The envirement has 12 observations for each frame
num_outputs = 1 # The envirement has 1 action space (flap or do nothing)
    
best_genome = load_best_genome("best_genome.txt")
w1, b1, w2, b2 = construct_nn(best_genome, input_size=num_inputs, output_size=num_outputs)
obs, _ = env.reset()
total_reward = 0
while True:
    
    
    
    # ===============================================================
    # Next action:
    # ===============================================================
    # (feed the observation to your agent here)
    # Action space:
    # 0 - do nothing
    # 1 - flap
    #action = env.action_space.sample()
    try:
        X = np.array(obs).reshape(1, 2)
    except:
        X = np.array([0.1,1.0]).reshape(1, 2)

        # Forward pass through the NN
    output = forward_pass(X, w1, b1, w2, b2)
    print(output)

    action = 1 if output > 0.7 else 0

    # Processing:
    obs, reward, terminated, info = env.step(action)
    #print(obs)
    # Rewards:
    # +0.1 - every frame it stays alive
    # +1.0 - successfully passing a pipe
    # -1.0 - dying
    # −0.5 - touch the top of the screen
    total_reward += reward
    
    env.render()
    time.sleep(1 / 30)  # FPS
    
    # Checking if the player is still alive
    if terminated:
        break
print("reward: ", total_reward)
    
  

env.close()