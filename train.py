from unityagents import UnityEnvironment
import numpy as np
import time
import itertools
import pickle
import sys
import torch
from collections import deque
import agents
from agents import AgentDQN, AgentPR
import matplotlib.pyplot as plt



def train_agent(agent, n_episodes=1000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995, beta0=0.6):
    """Train a Deep Q-Learning agent.
    
    Params
    ======
        agent (agent class)
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
        beta0 (float): Starting value of beta. Must be a value between 0 and 1. beta will linearly (by episodes) grow from beta0 to 1.
    """
    
    beta_inc = (1 - beta0)/n_episodes
    beta = beta0 
    print('Beta increment is', beta_inc)

    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start                    # initialize epsilon
    for i_episode in range(1, n_episodes+1):
        
        time_it = time.time()
        env_info = env.reset(train_mode=True)[brain_name] # reset the environment
        state = env_info.vector_observations[0]            # get the current state
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)
            env_info = env.step(action)[brain_name] 
            next_state = env_info.vector_observations[0]   # get the next state
            reward = env_info.rewards[0]                   # get the reward
            done = env_info.local_done[0]                  # see if episode has finished
            agent.step(state, action, reward, next_state, done, beta)
            state = next_state
            score += reward
            
            if done:
                break 
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        eps = max(eps_end, eps_decay*eps) # decrease epsilon
        beta += beta_inc # increase beta 
        time_taken = time.time() - time_it
        
        try:
            idx = agent.memory.memory_idx
        except:
            idx = ''
        print('\rEpisode {}\tAvg Score: {:.2f} \tEpisode time: {:.2f} s \tId {} \tMemory len: {} \t Current epsilon: {:.4f} \t Current beta: {:.4f} '.format(i_episode, np.mean(scores_window), time_taken, idx, len(agent.memory.memory), eps, beta), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if (np.mean(scores_window)>=15.0) or (i_episode == n_episodes):
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
            time_it = time.time()
            torch.save(agent.qnetwork_local.state_dict(), 
                       'results\checkpoint_{:.2f}_{}_{}.pth'.
                       format(np.mean(scores_window), i_episode, int(time_it))
                      )
            break
    return scores


def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


if __name__ == '__main__':

    if len(sys.argv) == 1: #A1:
        banana_path = sys.argv[1]
    else:    
        banana_path = r'C:\working\ngs-lib\udacity\deep reinforcement learning\deep-reinforcement-learning\p1_navigation_git\Banana_Windows_x86_64\Banana.exe'

    print(banana_path)

    env = UnityEnvironment(file_name=banana_path)

    
    
    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    
    env_info = env.reset(train_mode=True)[brain_name]
    
    # Action and state size
    action_size = brain.vector_action_space_size
    state = env_info.vector_observations[0]
    state_size = len(state)
    
    # Set parameters to grid-search over
    agent_type = ['PR'] # Available are: PR or DQN
    episodes = [1000] # number of episodes
    max_t = [1000] # max timesteps/actions per episode
    buffer_size = [int(10_000)] #  replay buffer size
    batch_size = [64]  # minibatch size
    gamma = [0.99] # discount factor
    tau = [1e-3] #[1e-3] # for soft update of target parameters
    lr = [5e-4] #, 5e-4] # learning rate 
    update_every = [4] # how often to update the network

    # Prioritized replay paramters - https://arxiv.org/abs/1511.05952
    alpha = [0.7] #0.5  # how to weigth the replay memory priority values pi_i, that is: P(i) = (p_i)^ALPHA / sum_k ( (p_k)^ALPHA )
    beta = [0.5] #, 0.7]  # how much the Q-updates should be updated according to the natural (correct) sampling distribution instead of the prioritized sampling distribution. Full compensation toward the natural distribution if beta = 1. Zero compensation if beta= 0.
    eps = [1e-5] # Positive value to avoid edge cases of transitions to never be visited

    all_scores = []
    parameter_list = [element for element in itertools.product(agent_type, episodes, max_t, buffer_size, batch_size, gamma, tau, lr, update_every, alpha, beta, eps)]

    for (agent_t, EPI, MAX_t, BUF, BATCH, GAM, TA, learning_rate, UPD, alp, bet, ep) in parameter_list:
        param = (EPI, MAX_t, BUF, BATCH, GAM, TA, learning_rate, UPD, alp, bet, ep)
        print('Training parameters')
        print('Episodes', EPI)
        print('Timesteps per episode', MAX_t)
        print('Buffer size (memory)', BUF, BATCH)
        print('Gamma (reward discount factor)', GAM)
        print('Tau (rate at which target network is updated)', TA)
        print('Learning rate (for neural network optimizer)', learning_rate)
        print('Update every (after how many steps to learn)', UPD)
        print('Weighting of priorities (prioritized replay)', alp)
        print('Weighting of importance sampled samples based on priorities (prioritized replay)', bet)
        print('Positive value to avoid non-exploration of some states', ep)
        #print(param)
        agents.BUFFER_SIZE = BUF   # replay buffer size
        agents.BATCH_SIZE = BATCH  # minibatch size
        agents.GAMMA = GAM         # discount factor
        agents.TAU = TA            # for soft update of target parameters
        agents.LR = learning_rate  # learning rate 
        agents.UPDATE_EVERY = UPD  # how often to update the network

        # Prioritized replay paramters - https://arxiv.org/abs/1511.05952
        agents.ALPHA = alp # how to weigth the replay memory priority values pi_i, that is: P(i) = (p_i)^ALPHA / sum_k ( (p_k)^ALPHA )
        BETA0 = bet # how much the Q-updates should be updated according to the natural (correct) sampling distribution instead of the prioritized sampling distribution. Full compensation toward the natural distribution if beta = 1. Zero compensation if beta= 0.
        agents.EPS = ep # Positive value to avoid edge cases of transitions to never be visited

        if agent_t == 'PR':
            agent = AgentPR(state_size=state_size, action_size=action_size, seed=0)
        elif agent_t == 'DQN':
            agent = AgentDQN(state_size=state_size, action_size=action_size, seed=0)

        scores = train_agent(agent=agent, n_episodes=EPI, max_t=MAX_t, beta0=BETA0)
        all_scores.append((param, scores))

    #available_cores = os.cpu_count() - 1
    
    # Save raw results 
    time_it = time.time()
    path = r'results\scores_all_{}.pickle'.format(int(time_it))
    pickle.dump(all_scores, open(path, 'wb'))
    
    # Plot the results 
    fig, ax = plt.subplots(figsize=(8, 4*len(all_scores)), nrows=len(all_scores))

    for i, (param, scores) in enumerate(all_scores):

        #ax[i] = fig.add_subplot(111)
        n = 100
        ax[i].plot(np.arange(len(scores)), scores, 'b')
        ax[i].plot(n+np.arange(len(moving_average(scores, n=n))), moving_average(scores, n=n), 'r')
        ax[i].set_ylabel('Score')
        ax[i].set_xlabel('Episode #')
        ax[i].set_title('Parameters: ' + str(param))
    
    figure_path = r'results\scores_all_plot_{}.png'.format(int(time_it))
    plt.suptitle('Evolution of score.')
    plt.tight_layout()
    fig.savefig(figure_path)
    
    plt.show()
    
    
    # env.close()

