import sys
import argparse
import metaPyAgent
import gym

##############################
# AGENT PARAMS:
# Fixed: num_actions, num_states(if applicable)
# Hyperparams: alpha, gamma, order (fourier), epsilon,
# run params: episodes, timesteps, trials
##############################

parser = argparse.ArgumentParser(description='run selected RL agent and function approximator on the supplied domain')
parser.add_argument('-algo', type=str, required=True, help='Should be an RL agent from the provided list of agents')
parser.add_argument('-fn', type=str, required=True, help='Should be a function approximation method from the provided list of function approximators')
args = parser.parse_args()

env = gym.make('MountainCar-v0')

def main():
    agent = metaPyAgent.agent()
    if args.algo=='sarsa':
        agent.run_sarsa(env)
    elif args.algo=='Q':
        agent.run_Q()

if __name__=="__main__":
    main()
