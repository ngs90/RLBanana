import numpy as np
import random
from collections import namedtuple, deque
import time
import matplotlib.pyplot as plt

from unityagents import UnityEnvironment

import torch
import torch.nn.functional as F
import torch.optim as optim 

from model import QNetwork
import sumtree as ST


# Model parameters
BUFFER_SIZE = int(10_000)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network

# Prioritized replay paramters - https://arxiv.org/abs/1511.05952
ALPHA = 0.7 # how to weigth the replay memory priority values pi_i, that is: P(i) = (p_i)^ALPHA / sum_k ( (p_k)^ALPHA )
#BETA = 1.0 # how much the Q-updates should be updated according to the natural (correct) sampling distribution instead of the prioritized sampling distribution. Full compensation toward the natural distribution if beta = 1. Zero compensation if beta= 0.
EPS = 1e-4 # Positive value to avoid edge cases of transitions to never be visited

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class AgentDQN():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBufferDQN(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
 

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                #experiences = self.memory.prioritized_sample(ALPHA, EPS)
                self.learn(experiences, GAMMA)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy()).astype(np.int32) # must be in32 to work with unity
        else:
            return random.choice(np.arange(self.action_size)).astype(np.int32) # must be in32 to work with unity

    def loss_function(self, y, y_hat):
        return torch.sum((y - y_hat).pow(2))
        
    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences
        
        # taget values
        qs_target = self.qnetwork_target(next_states).detach()
        qmax, qmax_index = torch.max(qs_target, axis=1)
        qmax = qmax.unsqueeze(1)
        #qmax = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        
        y = rewards + gamma * qmax * (1-dones)
        
        # current estimate 
        #y_all = self.qnetwork_local(states)         
        #y_hat = torch.Tensor([y_all[i][action] for i, action in enumerate(actions)]).unsqueeze(1)
        #print('y_hat v1', y_hat)
        
        y_hat = self.qnetwork_local(states).gather(1, actions)
        #print('y_hat v2', y_hat)
        
        # Set gradients to zero
        self.optimizer.zero_grad()
        # Calculate the loss between target and estimate 
        loss = self.loss_function(y, y_hat)
        # Backpropagation with loss function
        loss.backward()
        # Upate weights
        self.optimizer.step()
            
        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class ReplayBufferDQN:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        
        print(type(self.memory), len(self.memory), self.memory)
        
        experiences = np.random.choice(a=self.memory, size=self.batch_size, )

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


### DQN WITH PRIORITIZED REPLAY ###   


class AgentPR:
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBufferPR(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
    

    def priority(self, state, action, reward, next_state, done, gamma, alpha):
        """
        Description:
            Calculates priority of a given (state action, reward, next_state, done)-tuple.
            The priority if made non-zero given a positive constant EPS and the significance of the TD error is controlled with the alpha paramteter.
        
        Input:
            state:
            action:
            reward:
            next_state:
            done: 
            gamma: discount factor
            alpha: Weighting factor of the experience replays. I.e. how much should we care about the priorities. alpha=0: not to much. alpha=1: quite a bit.
        """
        
        # taget values
        state = torch.from_numpy(np.vstack([state])).float().to(device)
        action = torch.from_numpy(np.vstack([action])).long().to(device)
        reward = torch.from_numpy(np.vstack([reward])).float().to(device)
        next_state = torch.from_numpy(np.vstack([next_state])).float().to(device)
        done = torch.from_numpy(np.vstack([done]).astype(np.uint8)).float().to(device)
        
        qs_target = self.qnetwork_target(next_state).detach()
        qmax, qmax_index = torch.max(qs_target, axis=1)
        qmax = qmax.unsqueeze(1)
        y = reward + gamma * qmax * (1-done)
        y_hat = self.qnetwork_local(state).gather(1, action)
        # delta = TD error (used for prioritized replay)
        delta = y - y_hat
        return ((abs(delta) + EPS).detach() ** alpha).item()

    def step(self, state, action, reward, next_state, done, beta):

        
        # Calculate priority of the new sample 
        priority = self.priority(state, action, reward, next_state, done, GAMMA, ALPHA)
        
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done, priority)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.prioritized_sample()
                #experiences = self.memory.prioritized_sample(ALPHA, EPS)
                self.learn(experiences, GAMMA, ALPHA, beta)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy()).astype(np.int32) # must be in32 to work with unity
        else:
            return random.choice(np.arange(self.action_size)).astype(np.int32) # must be in32 to work with unity

    def loss_function(self, y, y_hat, imp_w):
        """
            imp_w: importance sampling weights for that sample
        """
        return torch.sum(imp_w*torch.clamp((y - y_hat).pow(2), 0, 1) ) # Following the clipping approach suggested in: https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf
        
    def learn(self, experiences, gamma, alpha, beta):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
            max_priority: Maximum priority given to any experience. Used to rescale the priorities to avoid outliers impacting heavly on the update. 
            beta: Bias correction of importance sampling weights. beta = 1 full bias correction. beta = 0 no bias correction.
        """
        states, actions, rewards, next_states, dones, priorities, idxs = experiences

        # taget values
        qs_target = self.qnetwork_target(next_states).detach()
        qmax, qmax_index = torch.max(qs_target, axis=1)
        qmax = qmax.unsqueeze(1)

        y = rewards + gamma * qmax * (1-dones)

        y_hat = self.qnetwork_local(states).gather(1, actions)

        # delta = TD error (used for prioritized replay)
        delta = y - y_hat
        
        # Importance sampling weights 
        if self.memory.max_priority:
            imp_w = (BUFFER_SIZE * priorities) ** (- beta) / self.memory.max_priority
            #imp_w = (priorities) ** (- BETA) / self.memory.max_priority
        else:
            imp_w = torch.Tensor([1])*len(y)
            
            
        imp_w = imp_w.to(device)
            
        # Set gradients to zero
        self.optimizer.zero_grad()
        # Calculate the loss between target and estimate 
        loss = self.loss_function(y, y_hat, imp_w) # # Adjust the loss according to the importance sampling distribution
        # Backpropagation with loss function
        loss.backward()
        # Update weights
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
            
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class ReplayBufferPR:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done", "priority", "idx"])
        self.seed = random.seed(seed)
        self.memory_idx = 0
        
        # Sum tree
        self.root_node = None
        self.leaf_nodes = None
        
        self.max_priority = None
        
        # helper functions (only for testing purposes)
        #self.statistics_counter = 0
        self.sum_priorities = []
        self.sum_all_priorities = []
    
    def add(self, state, action, reward, next_state, done, priority):
        """Add a new experience to memory."""
        self.memory_idx += 1
        e = self.experience(state, action, reward, next_state, done, priority, self.memory_idx)
        self.memory.append(e)
        
        # Having added it to the memory we also need to add it to the sum tree (but only if tree has already been created)
        if self.root_node is not None:
            
            # Determine which leaf to update
            leaf_idx = self.memory_idx % BUFFER_SIZE # TODO: check if indices match?
            #print('updating leaf', self.leaf_nodes[leaf_idx], 'value', self.leaf_nodes[leaf_idx].value)
            leaf = self.leaf_nodes[leaf_idx]
            # Update the leaf with an update priority
            ST.update(leaf, priority)
            #print('updated leaf', self.leaf_nodes[leaf_idx], 'value', self.leaf_nodes[leaf_idx].value)
      
    # Sum tree - https://github.com/Rakshith6/Sum-Tree
    # Sum trees https://www.fcodelabs.com/2019/03/18/Sum-Tree-Introduction/ 
    # Sum tree - https://adventuresinmachinelearning.com/sumtree-introduction-python/

    def sample_tree_idx(self):
        """
        Description:
            Samples self.batch_size number of samples from the sum tree given by self.root_node
        """
        tree_total = self.root_node.value
        iterations = self.batch_size
        tree_idxs = [ST.retrieve(rand_val, self.root_node).idx for rand_val in np.random.uniform(0, tree_total, size=iterations)]

        return tree_idxs
        
    @staticmethod
    def tree_idx_to_deque_idx(tree_idx, buffer_size, current_time):
        """
        Maps the id of the sum tree to the id of the deque. I.e. the id's of the deque keeps moving as the oldest gets deleted whereas the id's for the samples in the sum tree is stationary but the entries gets overwritten/updated to correspond with new experiences/samples.
        
        Input:
            tree_idx: id in the sum tree to convert to current deque id 
            buffer_size: the length of the deque list
            current_time: Current time to calculate the "offset" between the sum tree and deque id's
        Returns:
            deque_idx: the deque idx 
        """

        cur_time_mod_buf_size = current_time % buffer_size
        return buffer_size - (cur_time_mod_buf_size - tree_idx) if tree_idx < cur_time_mod_buf_size else tree_idx - cur_time_mod_buf_size
                
#         # Spelled out:
#         if tree_idx < cur_time_mod_buf_size: 
#             deque_idx = buffer_size - (cur_time_mod_buf_size - tree_idx)
#         else:
#             deque_idx = tree_idx - cur_time_mod_buf_size
#         return deque_idx
    
    def prioritized_sample(self):
        """Prioritized sample of a batch of experiences from memory."""
        
        if len(self.memory) < BUFFER_SIZE:
            experiences = random.sample(self.memory, k=self.batch_size)
        else:
            if self.root_node is None: # If no sum tree yet, create one (with fixed length of BUFFER_SIZE)
                print('')
                all_priorities = [e.priority for e in self.memory if e is not None] # = torch.from_numpy(np.vstack([e.priority for e in self.memory if e is not None])).float().to(device)
                print('Creating sum tree based on ', len(all_priorities))
                if len(all_priorities) % 2 != 0:
                    print('Uff! why is all priorities an uneven number?')
                self.max_priority = max(all_priorities)
                self.root_node, self.leaf_nodes = ST.create_tree(all_priorities, insertion_times=range(0, self.memory_idx))
                print('Sum tree created')
            
            tree_idxs = self.sample_tree_idx()
            deque_idxs = [self.tree_idx_to_deque_idx(tree_idx, BUFFER_SIZE, self.memory_idx) for tree_idx in tree_idxs]
            
            # Occasionally update the max priority
            if self.memory_idx % 3100 == 0: 
                #print('')
                #print('Updating max priority', self.max_priority)
                all_priorities = [e.priority for e in self.memory if e is not None]
                self.max_priority = max(all_priorities)
                #print('Update max priority', self.max_priority)
                #print('The first priorities', all_priorities[:30])
                
            
#             if self.memory_idx % 3100 == 0:
#                 print('Buffer size', BUFFER_SIZE)
#                 print('current memory idx ', self.memory_idx)
#                 print('current memory idx modulo buffer size', self.memory_idx % BUFFER_SIZE)
#                 print('tree_idxs', tree_idxs[:10])
#                 print('deque_idxs', deque_idxs[:10])


            experiences = [exp for i, exp in enumerate(self.memory) if i in deque_idxs]
    
            self.sum_priorities.append(np.mean([e.priority for e in experiences]))
            self.sum_all_priorities.append(np.mean([e.priority for e in self.memory]))
            if self.memory_idx % 30000 == 0: 
                print('')
                print('mean chosen priorities: \t {:.4f}'.format(np.mean(self.sum_priorities)))
                print('mean all priorities:    \t {:.4f}'.format(np.mean(self.sum_all_priorities)))
            
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        priorities = torch.from_numpy(np.vstack([e.priority for e in experiences if e is not None])).float().to(device)
        idxs = torch.from_numpy(np.vstack([e.idx for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones, priorities, idxs)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

    