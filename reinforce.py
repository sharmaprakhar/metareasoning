import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim

import env


class PolicyEstimator():
    def __init__(self):
        self.network = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
            nn.Softmax(dim=-1))

    def predict(self, state):
        return self.network(torch.FloatTensor(state))

def discount_rewards(rewards, gamma=0.99):
    reward = np.array([gamma**i * rewards[i] for i in range(len(rewards))])
    reward = reward[::-1].cumsum()[::-1]
    return reward - reward.mean()

def reinforce(env, policy_estimator, num_episodes=2000, batch_size=10, gamma=0.99):
    total_rewards = []
    batch_rewards = []
    batch_actions = []
    batch_states = []
    batch_counter = 1

    optimizer = optim.Adam(policy_estimator.network.parameters(), lr=0.001)

    action_space = np.arange(2)
    for episode in range(num_episodes):
        state = env.reset()

        states = []
        rewards = []
        actions = []

        is_episode_done = False

        while not is_episode_done:
            action_probs = policy_estimator.predict(state).detach().numpy()
            action = np.random.choice(action_space, p=action_probs)
            next_state, reward, is_episode_done = env.step(action)
            
            states.append(state)
            rewards.append(reward)
            actions.append(action)
            state = next_state

            if is_episode_done:
                batch_rewards.extend(discount_rewards(rewards, gamma))
                batch_states.extend(states)
                batch_actions.extend(actions)
                batch_counter += 1
                total_rewards.append(sum(rewards))

                if batch_counter == batch_size:
                    optimizer.zero_grad()
                    state_tensor = torch.FloatTensor(batch_states)
                    reward_tensor = torch.FloatTensor(batch_rewards)
                    action_tensor = torch.LongTensor(batch_actions)

                    logprob = torch.log(policy_estimator.predict(state_tensor))
                    selected_logprobs = reward_tensor * logprob[np.arange(len(action_tensor)), action_tensor]
                    loss = -selected_logprobs.mean()

                    loss.backward()

                    optimizer.step()

                    batch_rewards = []
                    batch_actions = []
                    batch_states = []
                    batch_counter = 1

                print("\rEp: {} Average of last 10 rewards: {:.2f}".format(episode + 1, np.mean(total_rewards[-10:])))

    return total_rewards


def main():
    print("Running REINFORCE...")

    metareasoning_env = env.Environment('problems/test.json', 200, 0.02, 3)
    policy_estimator = PolicyEstimator()
    rewards = reinforce(metareasoning_env, policy_estimator)

    window = 10
    smoothed_rewards = [np.mean(rewards[i-window:i+1]) if i > window else np.mean(rewards[:i+1]) for i in range(len(rewards))]

    plt.figure(figsize=(12, 8))
    plt.plot(rewards)
    plt.plot(smoothed_rewards)
    plt.ylabel('Total Rewards')
    plt.xlabel('Episodes')
    plt.show()


if __name__ == "__main__":
    main()
