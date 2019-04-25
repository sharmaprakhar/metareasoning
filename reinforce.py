import numpy as np
import matplotlib.pyplot as plt
import gym
import sys

import torch
from torch import nn
from torch import optim

import env

class policy_estimator():
    def __init__(self, env):
        self.n_inputs = env.observation_space.shape[0]
        self.n_outputs = env.action_space.n
        # self.n_inputs = 2
        # self.n_outputs = 2
        
        # Define network
        self.network = nn.Sequential(
            nn.Linear(self.n_inputs, 32),
            nn.ReLU(), 
            nn.Linear(32, self.n_outputs),
            nn.Softmax(dim=-1))
    
    def predict(self, state):
        action_probs = self.network(torch.FloatTensor(state))
        return action_probs

problem = 'problems/50-tsp.json'
# anytime_algorithm_env = env.Environment(problem, 200, 0.02, 3)
# s = np.asarray(anytime_algorithm_env.reset())
# pe = policy_estimator(anytime_algorithm_env)

r_env = gym.make('CartPole-v0')
s = r_env.reset()
pe = policy_estimator(r_env)


# print(pe.predict(s))
# print(pe.network(torch.FloatTensor(s)))



def discount_rewards(rewards, gamma=0.99):
    r = np.array([gamma**i * rewards[i] 
                  for i in range(len(rewards))])
    # Reverse the array direction for cumsum and then
    # revert back to the original order
    r = r[::-1].cumsum()[::-1]
    return r - r.mean()

def reinforce(env, policy_estimator, num_episodes=2000,
               batch_size=10, gamma=0.99):

    # Set up lists to hold results
    total_rewards = []
    batch_rewards = []
    batch_actions = []
    batch_states = []
    batch_counter = 1
    
    # Define optimizer
    optimizer = optim.Adam(policy_estimator.network.parameters(), 
                           lr=0.001)
    
    # action_space = np.arange(env.action_space.n)
    action_space = np.arange(2)
    for ep in range(num_episodes):
        s_0 = env.reset()
        states = []
        rewards = []
        actions = []
        complete = False
        while complete == False:
            # Get actions and convert to numpy array
            action_probs = policy_estimator.predict(s_0).detach().numpy()
            # print(action_probs)
            # print(action_space)
            action = np.random.choice(action_space, p=action_probs)
            # s_1, r, complete = env.step(action)
            s_1, r, complete, _ = env.step(action)
            
            states.append(s_0)
            rewards.append(r)
            actions.append(action)
            s_0 = s_1
            
            # If complete, batch data
            if complete:
                batch_rewards.extend(discount_rewards(rewards, gamma))
                batch_states.extend(states)
                batch_actions.extend(actions)
                batch_counter += 1
                total_rewards.append(sum(rewards))
                
                # If batch is complete, update network
                if batch_counter == batch_size:
                    optimizer.zero_grad()
                    state_tensor = torch.FloatTensor(batch_states)
                    # print(state_tensor.shape)
                    # print(state_tensor)
                    reward_tensor = torch.FloatTensor(batch_rewards)
                    # Actions are used as indices, must be LongTensor
                    action_tensor = torch.LongTensor(batch_actions)
                    
                    # Calculate loss
                    logprob = torch.log(
                        policy_estimator.predict(state_tensor))
                    selected_logprobs = reward_tensor * \
                        logprob[np.arange(len(action_tensor)), action_tensor]
                    loss = -selected_logprobs.mean()
                    
                    # Calculate gradients
                    loss.backward()
                    # Apply gradients
                    optimizer.step()
                    
                    batch_rewards = []
                    batch_actions = []
                    batch_states = []
                    batch_counter = 1
                    
                # Print running average
                # print("\rEp: {} Average of last 10 rewards: {:.2f}".format(ep + 1, np.mean(total_rewards[-10:])), end="")
                print("\rEp: {} Average of last 10 rewards: {:.2f}".format(ep + 1, np.mean(total_rewards[-10:])))
                
    return total_rewards

def run_reinforce():
    print("running reinforce...")
    # rewards = reinforce(anytime_algorithm_env, pe)
    rewards = reinforce(r_env, pe)
    window = 10
    smoothed_rewards = [np.mean(rewards[i-window:i+1]) if i > window 
                        else np.mean(rewards[:i+1]) for i in range(len(rewards))]
    
    plt.figure(figsize=(12,8))
    plt.plot(rewards)
    plt.plot(smoothed_rewards)
    plt.ylabel('Total Rewards')
    plt.xlabel('Episodes')
    plt.show() 

if __name__=="main":
    print("running reinforce...")
    # rewards = reinforce(anytime_algorithm_env, pe)
    rewards = reinforce(env, pe)
    window = 10
    smoothed_rewards = [np.mean(rewards[i-window:i+1]) if i > window 
                        else np.mean(rewards[:i+1]) for i in range(len(rewards))]
    
    plt.figure(figsize=(12,8))
    plt.plot(rewards)
    plt.plot(smoothed_rewards)
    plt.ylabel('Total Rewards')
    plt.xlabel('Episodes')
    plt.show() 