import numpy as np
import torch
import torch.nn as nn

import torch.nn.functional as F
import torch.optim as optim

from random import random, randrange

from warnings import simplefilter
from Model_Torch import DDDQN

device = torch.device("cpu")

simplefilter(action='ignore', category=FutureWarning)

loss_function = nn.MSELoss()

MAP_MAX_X = 20  # Width of the Map
MAP_MAX_Y = 8  # Height of the Map


# Deep Q Network off-policy
def log_output(output):
    output[output >= 15] = 15
    output[output <= -15] = -15
    return np.log10(output + 16)


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
            gamma=0.99,  # The discount factor
            epsilon=1,  # Epsilon - the exploration factor
            epsilon_min=0.05,  # The minimum epsilon
            epsilon_decay=0.999,  # The decay epislon for each update_epsilon time
            learning_rate=0.0001,  # The learning rate for the DQN network
            tau=0.125,  # The factor for updating the DQN target network from the DQN network
            use_cnn=False,
            use_capsule=False
    ):
        self.swamp_scores = [5, 20, 40, 100]
        self.data_index = 0
        self.action_space = action_space
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.tau = tau
        self.use_cnn = use_cnn
        self.use_capsule = use_capsule

        # self.model_1 = torch.load("Dense_Model_110000_6_38.pth", map_location='cpu').to(device)
        # self.model_2 = torch.load("TrainedModels/CNN_Model_15000.pth").to(device)

        # # Creating networks
        self.model_1 = self.create_model().to(device)  # Creating the DQN model
        self.model_2 = self.create_model().to(device)  # Creating the DQN model

        self.target_model_1 = self.create_model().to(device)  # Creating the DQN target model
        self.target_model_2 = self.create_model().to(device)

        self.optimizer1 = optim.Adam(self.model_1.parameters(), lr=learning_rate)
        self.optimizer2 = optim.Adam(self.model_2.parameters(), lr=learning_rate)

        self.optimizer1_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer1, gamma=1)
        self.optimizer2_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer2, gamma=1)

    def create_model(self):
        model = DDDQN(action_space=self.action_space)
        return model

    def train_step1(self, state, dead, gold, energy, player, action):
        self.model_1.train()
        self.optimizer1.zero_grad()

        state = torch.from_numpy(state).float().to(device)
        dead = torch.from_numpy(dead).float().to(device)
        gold = torch.from_numpy(gold).float().to(device)
        energy = torch.from_numpy(energy).float().to(device)
        player = torch.from_numpy(player).float().to(device)
        action = torch.from_numpy(action).float().to(device)

        action_predict = self.model_1(state, dead, gold, energy, player)
        loss = loss_function(action_predict, action)

        loss.backward()
        self.optimizer1.step()

        return loss.detach().cpu().numpy()

    def train_step2(self, state, dead, gold, energy, player, action):
        self.model_2.train()
        self.optimizer2.zero_grad()

        state = torch.from_numpy(state).float().to(device)
        dead = torch.from_numpy(dead).float().to(device)
        gold = torch.from_numpy(gold).float().to(device)
        energy = torch.from_numpy(energy).float().to(device)
        player = torch.from_numpy(player).float().to(device)
        action = torch.from_numpy(action).float().to(device)

        action_predict = self.model_2(state, dead, gold, energy, player)
        loss = loss_function(action_predict, action)

        loss.backward()
        self.optimizer2.step()

        return loss.detach().cpu().numpy()

    def act_1(self, state, dead, gold, energy, player):
        state = torch.from_numpy(state).float().to(device)
        dead = torch.from_numpy(dead).float().to(device)
        gold = torch.from_numpy(gold).float().to(device)
        energy = torch.from_numpy(energy).float().to(device)
        player = torch.from_numpy(player).float().to(device)
        self.model_1.eval()
        return self.model_1.advantage(state, dead, gold, energy, player).detach().cpu().numpy()

    def act_2(self, state, dead, gold, energy, player):
        state = torch.from_numpy(state).float().to(device)
        dead = torch.from_numpy(dead).float().to(device)
        gold = torch.from_numpy(gold).float().to(device)
        energy = torch.from_numpy(energy).float().to(device)
        player = torch.from_numpy(player).float().to(device)
        self.model_2.eval()
        return self.model_2.advantage(state, dead, gold, energy, player).detach().cpu().numpy()

    def act(self, state, dead, gold, energy, player, finish_object, act_epsisode, finish_object_no_gold):
        a_maxes = epsilon_greedy_policy(self.act_1(state, dead, gold, energy, player))

        if energy[0, 0] * 50 <= 5:
            return [4, a_maxes[0][0]]

        if random() < self.epsilon:
            a_chosen = randrange(self.action_space)
            real_a_chosen = a_chosen
            if finish_object['0'] and finish_object['1'] and finish_object['2'] and finish_object['3'] and finish_object['4'] and finish_object['5']:
                print('finish_object_no_gold ', finish_object_no_gold)
                print('finish_object_no_gold ', finish_object_no_gold)
                print('finish_object_no_gold ', finish_object_no_gold)
                if finish_object['0'] and finish_object['1'] and finish_object['2'] and finish_object['3'] and finish_object['4'] and finish_object['5']:
                    actions = ['0', '1', '2', '3']
                    _index = 0
                    while finish_object_no_gold[actions[_index]] and _index <= 3:
                        _index += 1
                        if _index > 3:
                            break
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

    def relay_1(self, samples, batch_size):
        inputs_state = np.zeros((batch_size, 25, 6))
        inputs_dead = np.zeros((batch_size, 6))
        inputs_gold = np.zeros((batch_size, 4))
        inputs_energy = np.zeros((batch_size, 100))
        inputs_player = np.zeros((batch_size, 12))

        targets = np.zeros((batch_size, self.action_space))

        self.target_model_1.eval()
        self.target_model_2.eval()

        for i in range(0, batch_size):
            state          = samples[0][i, :, :]
            state_dead     = samples[1][i, :]
            state_gold     = samples[2][i, :]
            state_en       = samples[3][i, :]
            state_player   = samples[4][i, :]
            action         = samples[5][i]
            real_action    = samples[6][i]
            reward         = samples[7][i]
            new_state      = samples[8][i, :, :]
            new_state_dead = samples[9][i, :]
            new_state_gold = samples[10][i, :]
            new_state_en   = samples[11][i, :]
            new_state_player = samples[12][i, :]
            done           = samples[13][i]

            inputs_state[i, :, :] = state
            inputs_dead[i, :] = state_dead
            inputs_gold[i, :] = state_gold
            inputs_energy[i, :] = state_en
            inputs_player[i, :] = state_player

            targets[i, :] = self.target_model_1(
                torch.from_numpy(np.expand_dims(state, 0)).float().to(device),
                torch.from_numpy(np.expand_dims(state_dead, 0)).float().to(device),
                torch.from_numpy(np.expand_dims(state_gold, 0)).float().to(device),
                torch.from_numpy(np.expand_dims(state_en, 0)).float().to(device),
                torch.from_numpy(np.expand_dims(state_player, 0)).float().to(device)
            ).detach().cpu().numpy()

            if action != real_action:
                action = real_action
                targets[i, action] = -15
            else:
                if done:
                    targets[i, action] = reward  # if terminated, only equals reward
                else:
                    q_future = np.max(self.target_model_1(
                        torch.from_numpy(np.expand_dims(new_state, 0)).float().to(device),
                        torch.from_numpy(np.expand_dims(new_state_dead, 0)).float().to(device),
                        torch.from_numpy(np.expand_dims(new_state_gold, 0)).float().to(device),
                        torch.from_numpy(np.expand_dims(new_state_en, 0)).float().to(device),
                        torch.from_numpy(np.expand_dims(new_state_player, 0)).float().to(device)
                    ).detach().cpu().numpy())
                    targets[i, action] = reward + q_future * self.gamma

        loss = self.train_step1(inputs_state, inputs_dead, inputs_gold, inputs_energy, inputs_player, log_output(targets))

        self.optimizer1_scheduler.step()

        return loss

    def relay_2(self, samples, batch_size):
        inputs_state = np.zeros((batch_size, 25, 6))
        inputs_dead = np.zeros((batch_size, 6))
        inputs_gold = np.zeros((batch_size, 4))
        inputs_energy = np.zeros((batch_size, 100))
        inputs_player = np.zeros((batch_size, 12))

        targets = np.zeros((batch_size, self.action_space))

        self.target_model_1.eval()
        self.target_model_2.eval()

        for i in range(0, batch_size):
            state = samples[0][i, :, :]
            state_dead = samples[1][i, :]
            state_gold = samples[2][i, :]
            state_en = samples[3][i, :]
            state_player = samples[4][i, :]
            action = samples[5][i]
            real_action = samples[6][i]
            reward = samples[7][i]
            new_state = samples[8][i, :, :]
            new_state_dead = samples[9][i, :]
            new_state_gold = samples[10][i, :]
            new_state_en = samples[11][i, :]
            new_state_player = samples[12][i, :]
            done = samples[13][i]

            inputs_state[i, :, :] = state
            inputs_dead[i, :] = state_dead
            inputs_gold[i, :] = state_gold
            inputs_energy[i, :] = state_en
            inputs_player[i, :] = state_player

            targets[i, :] = self.target_model_1(
                torch.from_numpy(np.expand_dims(state, 0)).float().to(device),
                torch.from_numpy(np.expand_dims(state_dead, 0)).float().to(device),
                torch.from_numpy(np.expand_dims(state_gold, 0)).float().to(device),
                torch.from_numpy(np.expand_dims(state_en, 0)).float().to(device),
                torch.from_numpy(np.expand_dims(state_player, 0)).float().to(device)
            ).detach().cpu().numpy()

            if action != real_action:
                action = real_action
                targets[i, action] = -15
            else:
                if done:
                    targets[i, action] = reward  # if terminated, only equals reward
                else:
                    q_future = np.max(self.target_model_1(
                        torch.from_numpy(np.expand_dims(new_state, 0)).float().to(device),
                        torch.from_numpy(np.expand_dims(new_state_dead, 0)).float().to(device),
                        torch.from_numpy(np.expand_dims(new_state_gold, 0)).float().to(device),
                        torch.from_numpy(np.expand_dims(new_state_en, 0)).float().to(device),
                        torch.from_numpy(np.expand_dims(new_state_player, 0)).float().to(device)
                    ).detach().cpu().numpy())
                    targets[i, action] = reward + q_future * self.gamma

        loss = self.train_step2(inputs_state, inputs_dead, inputs_gold, inputs_energy, inputs_player,
                                log_output(targets))

        self.optimizer2_scheduler.step()

        return loss

    def replay(self, samples, batch_size):
        p = np.random.random()
        # if p < .5:
        return self.relay_1(samples, batch_size)
        # else:
        #     return self.relay_2(samples, batch_size)

    def target_train_1(self):
        for target_param, param in zip(self.target_model_1.parameters(), self.model_1.parameters()):
            target_param.data.copy_(self.tau * param.data + target_param.data * (1.0 - self.tau))

    def target_train_2(self):
        for target_param, param in zip(self.target_model_2.parameters(), self.model_2.parameters()):
            target_param.data.copy_(self.tau * param.data + target_param.data * (1.0 - self.tau))

    def target_train(self):
        self.target_train_1()
        # self.target_train_2()

    def update_epsilon(self, epsilon):
        self.epsilon = epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon)

    def save_model_1(self, path, model_name):
        torch.save(self.model_1, "TrainedModels/Dense_Model_1" + path + ".pth")

    def save_model_2(self, path, model_name):
        torch.save(self.model_2, "TrainedModels/CNN_Model_1" + path + ".pth")

    def save_model(self, path, model_name):
        # serialize model to JSON
        self.save_model_1(path, model_name)
        # self.save_model_2(path, model_name)
