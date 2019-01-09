##############################################################################
# Main module for metaPy
# Author: Prakhar Sharma
##############################################################################
import sys
import argparse
# import metaPyAgent
import mpyAg_test
#import gym
import en
import tab_q
##############################
# AGENT PARAMS:
# Fixed: num_actions, num_states(if applicable)
# Hyperparams: alpha, gamma, order (fourier), epsilon,
# run params: episodes, timesteps, trials
##############################

parser = argparse.ArgumentParser(description='run selected RL agent and function approximator on the supplied domain')
parser.add_argument('-algo', type=str, required=True, help='Should be an RL agent from the provided list of agents')
parser.add_argument('-fn', type=str, required=False, help='Should be a function approximation method from the provided list of function approximators')
parser.add_argument('-ep', type=float, required=True, help='prob for epsilon greedy policy')
parser.add_argument('-alpha', type=float, required=True, help='learning rate for parameter update')
parser.add_argument('-gamma', type=float, required=True, help='discount factor')
parser.add_argument('-order', type=int, required=True, help='fourier basis order')
parser.add_argument('-episodes', type=int, required=True, help='number of epiosdes to be run')
parser.add_argument('-trials', type=int, required=True, help='number of trials')
parser.add_argument('-ti', type=int, required=False, help='number of timesteps')

#trials and timesteps are optional because they both can be extracted from the dataset
args = parser.parse_args()
params= {}
params['epsilon'] = args.ep
params['gamma'] = args.gamma
params['order'] = args.order
params['episodes'] = args.episodes
params['trials'] = args.trials
params['timesteps'] = args.ti
params['actions'] = [0,1]
params['alpha'] = args.alpha


def main():
    performance_file = '../simulations/50-tsp-0.1s.json'
    e = en.env(performance_file)
    agent = mpyAg_test.agent(params)
    baseline = tab_q.tabQ(params)

    baseline.run_tabSarsa(e)
    import sys
    sys.exit()

    if args.algo=='sarsa':
        agent.run_sarsa(e)
    elif args.algo=='Q':
    	agent.run_Q(e)
    elif args.algo=='nac-lstd':
    	agent.run_nac_lstd(e)

if __name__=="__main__":
    main()
