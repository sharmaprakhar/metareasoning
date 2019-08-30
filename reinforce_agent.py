import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim

import env


class Agent():
    def __init__(self, params, env):
        self.params = params
        self.env = env
        self.network = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
            nn.Softmax(dim=-1))

    def get_action(self, state):
        action_probabilities = self.get_prediction(state).detach().numpy()
        return np.random.choice(self.env.ACTIONS, p=action_probabilities)

    def get_prediction(self, state):
        return self.network(torch.FloatTensor(state))

    def get_loss(self, batch_states, batch_actions, batch_rewards):
        state_tensor = torch.FloatTensor(batch_states)
        action_tensor = torch.LongTensor(batch_actions)
        reward_tensor = torch.FloatTensor(batch_rewards)

        log_probabilities = torch.log(self.get_prediction(state_tensor))
        selected_log_probabilities = reward_tensor * log_probabilities[np.arange(len(action_tensor)), action_tensor]

        return -selected_log_probabilities.mean()

    def get_discounted_rewards(self, rewards):
        reward = np.array([self.params["gamma"]**i * rewards[i] for i in range(len(rewards))])
        reward = reward[::-1].cumsum()[::-1]
        return reward - reward.mean()

    def run_reinforce(self, statistics):
        batch_states = []
        batch_actions = []
        batch_rewards = []
        batch_counter = 1

        optimizer = optim.Adam(self.network.parameters(), lr=self.params["learning_rate"])

        for _ in range(self.params["episodes"]):
            state = self.env.reset()
            action = self.get_action(state)

            states = []
            actions = []
            rewards = []

            while True:
                next_state, reward, is_episode_done = self.env.step(action)

                states.append(state)
                actions.append(action)
                rewards.append(reward)

                next_action = self.get_action(state)

                if is_episode_done:
                    batch_states.extend(states)
                    batch_actions.extend(actions)
                    batch_rewards.extend(self.get_discounted_rewards(rewards))
                    batch_counter += 1

                    if batch_counter == self.params["batch_size"]:
                        optimizer.zero_grad()

                        loss = self.get_loss(batch_states, batch_actions, batch_rewards)
                        loss.backward()

                        optimizer.step()

                        batch_states = []
                        batch_actions = []
                        batch_rewards = []
                        batch_counter = 1

                    statistics["stopping_points"].append(next_state[1])
                    statistics["utilities"].append(self.env.get_utility())

                    break

                state = next_state
                action = next_action


def main():
    params = {
        "episodes": 2000,
        "batch_size": 10,
        "gamma": 1,
        "learning_rate": 0.01
    }
    metareasoning_env = env.Environment('problems/test.json', 200, 0.3, 1)
    agent = Agent(params, metareasoning_env)

    statistics = {"stopping_points": [], "utilities": []}
    agent.run_reinforce(statistics)


if __name__ == "__main__":
    main()
