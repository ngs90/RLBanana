from unityagents import UnityEnvironment
import numpy as np
import time
import itertools
import pickle
import sys
import torch
from collections import deque
from agent import AgentDQN, AgentPR
import matplotlib.pyplot as plt


#def test_agent():
    
if __name__ == '__main__':

    if len(sys.argv) == 3: #A1:
        banana_path = sys.argv[1]
        state_dict_path = sys.argv[2]
    else:    
        banana_path = r'C:\working\ngs-lib\udacity\deep reinforcement learning\deep-reinforcement-learning\p1_navigation\Banana_Windows_x86_64\Banana.exe'
        state_dict_path = r'results\good_model.pth'
    print(banana_path)

    env = UnityEnvironment(file_name=banana_path)

    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    
    # Initialize in train mode
    env_info = env.reset(train_mode=True)[brain_name]
    
    # Action and state size
    action_size = brain.vector_action_space_size
    state = env_info.vector_observations[0]
    state_size = len(state)
    
    
    
    agent_t = 'PR'
    # Initialize fresh agent
    if agent_t == 'PR':
        agent = AgentPR(state_size=state_size, action_size=action_size, seed=0)
    elif agent_t == 'DQN':
        agent = AgentDQN(state_size=state_size, action_size=action_size, seed=0)
    
    # Load good model weights 
    
    state_dict = torch.load(state_dict_path)
    # Insert model weights in agent 
    agent.qnetwork_local.load_state_dict(state_dict)

    agent.qnetwork_local.eval() # Remember that you must call model.eval() to set dropout and batch normalization layers to evaluation mode before running inference. Failing to do this will yield inconsistent inference results.
    
    # Put in test mode
    env_info = env.reset(train_mode=False)[brain_name] 
    state = env_info.vector_observations[0]            # get the current state
    score = 0                                          # initialize the score
    while True:
        action = agent.act(state, 0)
        env_info = env.step(action)[brain_name] 
        next_state = env_info.vector_observations[0]   # get the next state
        reward = env_info.rewards[0]                   # get the reward
        done = env_info.local_done[0]                  # see if episode has finished
        #agent.step(state, action, reward, next_state, done)
        state = next_state
        score += reward
        if done:                                       # exit loop if episode finished
            break
    
    print("Score: {}".format(score))
    
    env.close()