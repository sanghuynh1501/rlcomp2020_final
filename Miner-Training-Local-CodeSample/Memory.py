import numpy as np
import random


class Memory:
    capacity = None

    def __init__(
            self,
            capacity,
            length=None,
            states=None,
            state_deads=None,
            state_golds=None,
            state_players=None,
            state_gens=None,
            actions=None,
            real_actions=None,
            rewards=None,
            dones=None,
            states2=None,
            state_golds2=None,
            state_gens2=None,
            state_players2=None
    ):
        self.capacity = capacity
        self.length = 0
        self.states = states
        self.state_deads = state_deads
        self.state_golds = state_golds
        self.state_gens = state_gens
        self.state_players = state_players
        self.actions = actions
        self.real_actions = real_actions
        self.rewards = rewards
        self.dones = dones
        self.states2 = states2
        self.state_deads2 = state_deads
        self.state_golds2 = state_golds2
        self.state_gens2 = state_gens2
        self.state_players2 = state_players2

    def push(self, s, s_dead, s_gold, s_gen, s_player, a, r_a, r, done, s2, s_dead2, s_gold2, s_gen2, s_player2):
        if self.states is None:
            self.states = s
            self.state_deads = s_dead
            self.state_golds = s_gold
            self.state_gens = s_gen
            self.state_players = s_player
            self.actions = a
            self.real_actions = r_a
            self.rewards = r
            self.dones = done
            self.states2 = s2
            self.state_deads2 = s_dead2
            self.state_golds2 = s_gold2
            self.state_gens2 = s_gen2
            self.state_players2 = s_player2
        else:
            self.states = np.vstack((self.states, s))
            self.state_deads = np.vstack((self.state_deads, s_dead))
            self.state_golds = np.vstack((self.state_golds, s_gold))
            self.state_gens = np.vstack((self.state_gens, s_gen))
            self.state_players = np.vstack((self.state_players, s_player))
            self.actions = np.vstack((self.actions, a))
            self.real_actions = np.vstack((self.real_actions, r_a))
            self.rewards = np.vstack((self.rewards, r))
            self.dones = np.vstack((self.dones, done))
            self.states2 = np.vstack((self.states2, s2))
            self.state_deads2 = np.vstack((self.state_deads2, s_dead2))
            self.state_golds2 = np.vstack((self.state_golds2, s_gold2))
            self.state_gens2 = np.vstack((self.state_gens2, s_gen2))
            self.state_players2 = np.vstack((self.state_players2, s_player2))
        self.length = self.length + 1

        if self.length > self.capacity:
            self.states = np.delete(self.states, 0, axis=0)
            self.state_deads = np.delete(self.state_deads, 0, axis=0)
            self.state_golds = np.delete(self.state_golds, 0, axis=0)
            self.state_gens = np.delete(self.state_gens, 0, axis=0)
            self.state_players = np.delete(self.state_players, 0, axis=0)
            self.actions = np.delete(self.actions, 0, axis=0)
            self.real_actions = np.delete(self.real_actions, 0, axis=0)
            self.rewards = np.delete(self.rewards, 0, axis=0)
            self.dones = np.delete(self.dones, 0, axis=0)
            self.states2 = np.delete(self.states2, 0, axis=0)
            self.state_deads2 = np.delete(self.state_deads2, 0, axis=0)
            self.state_golds2 = np.delete(self.state_golds2, 0, axis=0)
            self.state_gens2 = np.delete(self.state_gens2, 0, axis=0)
            self.state_players2 = np.delete(self.state_players2, 0, axis=0)
            self.length = self.length - 1

    def sample(self, batch_size):
        if self.length >= batch_size:
            idx = random.sample(range(0, self.length), batch_size)
            s = self.states[idx, :]
            s_dead = self.state_deads[idx, :]
            s_gold = self.state_golds[idx, :]
            s_gen = self.state_gens[idx, :]
            s_player = self.state_players[idx, :]
            a = self.actions[idx, :]
            r_a = self.real_actions[idx, :]
            r = self.rewards[idx, :]
            d = self.dones[idx, :]
            s2 = self.states2[idx, :]
            s_dead2 = self.state_deads2[idx, :]
            s_gold2 = self.state_golds2[idx, :]
            s_gen2 = self.state_gens2[idx, :]
            s_player2 = self.state_players2[idx, :]

            return list([s, s_dead, s_gold, s_gen, s_player, a, r_a, r, s2, s_dead2, s_gold2, s_gen2, s_player2, d])
