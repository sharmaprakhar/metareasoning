import numpy as np
import torch
from torch import nn, optim


class Agent():
    NUM_INPUTS = 2
    NUM_OUTPUTS = 2
    BATCH_SIZE = 10

    def __init__(self, params, env):
        self.params = params
        self.env = env
        self.policy = nn.Sequential(
            nn.Linear(self.NUM_INPUTS, 16),
            nn.ReLU(),
            nn.Linear(16, self.NUM_OUTPUTS),
            nn.Softmax(dim=-1)
        )

    def run_reinforce(self, statistics):
        batch_states = []
        batch_actions = []
        batch_rewards = []
        batch_counter = 1

        optimizer = optim.Adam(self.policy.parameters(), lr=self.params["alpha"])

        for _ in range(self.params["episodes"]):
            states = []
            rewards = []
            actions = []

            state = self.env.reset()

            while True:
                action_probilities = self.policy(torch.FloatTensor(state)).detach().numpy()
                action = np.random.choice(self.env.ACTIONS, p=action_probilities)

                next_state, reward, is_episode_done = self.env.step(action)

                states.append(state)
                rewards.append(reward)
                actions.append(action)

                state = next_state

                if is_episode_done:
                    batch_states.extend(states)
                    batch_actions.extend(actions)
                    batch_rewards.extend(discount_rewards(rewards, self.params["gamma"]))
                    batch_counter += 1

                    if batch_counter == self.BATCH_SIZE:
                        optimizer.zero_grad()

                        state_tensor = torch.FloatTensor(batch_states)
                        action_tensor = torch.LongTensor(batch_actions)
                        reward_tensor = torch.FloatTensor(batch_rewards)

                        log_probabilities = torch.log(self.policy(state_tensor))
                        selected_log_probabilities = reward_tensor * log_probabilities[np.arange(len(action_tensor)), action_tensor]
                        loss = -selected_log_probabilities.mean()
                        loss.backward()

                        optimizer.step()

                        batch_states = []
                        batch_actions = []
                        batch_rewards = []
                        batch_counter = 1

                    utility = self.env.get_utility()
                    optimal_utility = self.env.get_optimal_utility()
                    error = abs((utility - optimal_utility) / optimal_utility)

                    statistics["errors"].append(error)
                    statistics["smoothed_errors"].append(np.average(statistics["errors"][-50:]))
                    statistics["stopping_points"].append(next_state[1])
                    statistics["smoothed_stopping_points"].append(np.average(statistics["stopping_points"][-50:]))

                    break


def discount_rewards(rewards, gamma):
    reward = np.array([gamma**time * reward for time, reward in enumerate(rewards)])
    reward = reward[::-1].cumsum()[::-1]
    return reward - reward.mean()
