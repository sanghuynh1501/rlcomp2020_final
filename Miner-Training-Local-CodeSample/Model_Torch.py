import torch

import torch.nn as nn
import torch.nn.functional as F


class DDDQN(nn.Module):

    def __init__(self, action_space):
        super(DDDQN, self).__init__()

        self.layer1 = nn.Linear(6, 32)
        self.layer2 = nn.Linear(32, 64)
        self.flatten_reduce = nn.Linear(5 * 5 * 64, 128)

        self.dead_layer1 = nn.Linear(6, 64)
        self.dead_layer2 = nn.Linear(64, 128)

        self.gold_layer1 = nn.Linear(4, 64)
        self.gold_layer2 = nn.Linear(64, 128)

        self.energy_layer1 = nn.Linear(100, 64)
        self.energy_layer2 = nn.Linear(64, 128)

        self.player_layer1 = nn.Linear(12, 64)
        self.player_layer2 = nn.Linear(64, 128)

        self.concat_layer1 = nn.Linear(128 * 5, 256)
        self.concat_layer2 = nn.Linear(256, 128)
        self.v = nn.Linear(128, 1)
        self.a = nn.Linear(128, action_space)

    def forward(self, state, dead, gold, energy, player):
        state = F.relu(self.layer1(state))
        state = F.relu(self.layer2(state))
        state = state.view(-1, 5 * 5 * 64)
        state = F.relu(self.flatten_reduce(state))

        dead = F.relu(self.dead_layer1(dead))
        dead = dead.view(-1, 64)
        dead = F.relu(self.dead_layer2(dead))

        gold = F.relu(self.gold_layer1(gold))
        gold = gold.view(-1, 64)
        gold = F.relu(self.gold_layer2(gold))

        energy = F.relu(self.energy_layer1(energy))
        energy = F.relu(self.energy_layer2(energy))

        player = F.relu(self.player_layer1(player))
        player = F.relu(self.player_layer2(player))

        concat = torch.cat((state, dead, gold, energy, player), dim=-1)
        concat = self.concat_layer1(concat)
        concat = self.concat_layer2(concat)

        v = self.v(concat)
        a = self.a(concat)

        Q = v + (a - torch.mean(a, 1, keepdim=True))

        return Q

    def advantage(self, state, dead, gold, energy, player):
        state = F.relu(self.layer1(state))
        state = F.relu(self.layer2(state))
        state = state.view(-1, 5 * 5 * 64)
        state = F.relu(self.flatten_reduce(state))

        dead = F.relu(self.dead_layer1(dead))
        dead = dead.view(-1, 64)
        dead = F.relu(self.dead_layer2(dead))

        gold = F.relu(self.gold_layer1(gold))
        gold = gold.view(-1, 64)
        gold = F.relu(self.gold_layer2(gold))

        energy = F.relu(self.energy_layer1(energy))
        energy = F.relu(self.energy_layer2(energy))

        player = F.relu(self.player_layer1(player))
        player = F.relu(self.player_layer2(player))

        concat = torch.cat((state, dead, gold, energy, player), dim=-1)
        concat = self.concat_layer1(concat)
        concat = self.concat_layer2(concat)

        a = self.a(concat)

        return a
