import numpy as np
import torch
import torch.nn as nn

import torch.nn.functional as F
import torch.optim as optim

from random import random, randrange

from warnings import simplefilter

device = torch.device("cpu")

simplefilter(action='ignore', category=FutureWarning)

MAP_MAX_X = 20  # Width of the Map
MAP_MAX_Y = 8  # Height of the Map


def epsilon_greedy_policy(
        q_values,
):
    greedy_action_indicies_0 = np.argmax(q_values, axis=1)
    greedy_action_indicies_1 = np.array([np.argsort(np.max(q_values, axis=0))[-2]])
    greedy_action_indicies_2 = np.array([np.argsort(np.max(q_values, axis=0))[-3]])
    greedy_action_indicies_3 = np.array([np.argsort(np.max(q_values, axis=0))[-4]])
    greedy_action_indicies_4 = np.array([np.argsort(np.max(q_values, axis=0))[-5]])
    greedy_action_indicies_5 = np.array([np.argsort(np.max(q_values, axis=0))[-6]])

    return [
        greedy_action_indicies_0,
        greedy_action_indicies_1,
        greedy_action_indicies_2,
        greedy_action_indicies_3,
        greedy_action_indicies_4,
        greedy_action_indicies_5
    ]


class DQN:

    def __init__(
            self,
            action_space,  # The number of actions for the DQN network
            epsilon=0.05
    ):
        self.action_space = action_space
        self.epsilon = epsilon

        self.model_1 = torch.load("Dense_Model_110000_6_38.pth").to(device)

    def act_1(self, state, dead, gold, energy, player):
        state = torch.from_numpy(state).float().to(device)
        dead = torch.from_numpy(dead).float().to(device)
        gold = torch.from_numpy(gold).float().to(device)
        energy = torch.from_numpy(energy).float().to(device)
        player = torch.from_numpy(player).float().to(device)
        self.model_1.eval()
        return self.model_1.advantage(state, dead, gold, energy, player).detach().cpu().numpy()

    def act(self, state, dead, gold, energy, player, finish_object, act_epsisode, finish_object_no_gold):
        a_maxes = epsilon_greedy_policy(
            self.act_1(state, dead, gold, energy, player))

        if energy[0, 0] * 50 <= 7:
            return [4, a_maxes[0][0]]

        if random() < self.epsilon:
            a_chosen = randrange(self.action_space)
            real_a_chosen = a_chosen
            if finish_object['0'] and finish_object['1'] and finish_object['2'] and finish_object['3'] and \
                    finish_object['4']:
                if finish_object['0'] and finish_object['1'] and finish_object['2'] and finish_object['3'] and \
                        finish_object['4']:
                    actions = ['0', '1', '2', '3']
                    _index = 0
                    while finish_object_no_gold[actions[_index]] and _index <= 3:
                        _index += 1
                    if _index > 3:
                        return [4, a_maxes[_index][0]]
                    return [int(actions[_index]), real_a_chosen]
            while finish_object[str(a_chosen)]:
                a_chosen = randrange(self.action_space)
            return [a_chosen, real_a_chosen]
        else:
            if finish_object['0'] and finish_object['1'] and finish_object['2'] and finish_object['3'] and \
                    finish_object['4'] and finish_object['5']:
                _index = 0
                a_chosen = a_maxes[_index][0]
                finish_object_no_gold['4'] = False
                finish_object_no_gold['5'] = False
                while finish_object_no_gold[str(a_chosen)]:
                    _index += 1
                    if _index < 4:
                        a_chosen = a_maxes[_index][0]
                    else:
                        print("randrange(self.action_space) ", randrange(self.action_space))
                        a_chosen = randrange(self.action_space)
            else:
                index = 0
                a_chosen = a_maxes[index][0]
                while finish_object[str(a_chosen)]:
                    index += 1
                    if index < 6:
                        a_chosen = a_maxes[index][0]
                    else:
                        print("randrange(self.action_space) ", randrange(self.action_space))
                        a_chosen = randrange(self.action_space)
        return [a_chosen, a_maxes[0][0]]