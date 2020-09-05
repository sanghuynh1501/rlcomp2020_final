import sys
import matplotlib.pyplot as plt

from MINER_STATE import State
from DQNModel_Torch import DQN  # A class of creating a deep q-learning model
from MinerEnv import MinerEnv
from Memory import Memory  # A class of creating a batch in order to store experiences for the training process

import pandas as pd
import datetime
import numpy as np

HOST = "localhost"
PORT = 1111
if len(sys.argv) == 3:
    HOST = str(sys.argv[1])
    PORT = int(sys.argv[2])

now = datetime.datetime.now()
header = ["Ep", "Step", "Reward", "Total_reward", "Action", "Epsilon", "Done", "Termination_Code"]
filename = "Data/data_" + now.strftime("%Y%m%d-%H%M") + ".csv"
with open(filename, 'w') as f:
    pd.DataFrame(columns=header).to_csv(f, encoding='utf-8', index=False, header=True)

MEAN_STEP = 100
MAX_STEP = 1000  # The number of steps for each episode
BATCH_SIZE = 32  # The number of experiences for each replay
N_EPISODE = 10000  # The number of episodes for training
SAVE_NETWORK = 100  # After this number of episodes, the DQN model is saved for testing later.
MEMORY_SIZE = 100000  # The size of the batch for storing experiences
INITIAL_REPLAY_SIZE = 1000  # The number of experiences are stored in the memory batch before starting replaying
ACTIONNUM = 6  # The number of actions output from the DQN model
MAP_MAX_X = 21  # Width of the Map
MAP_MAX_Y = 9  # Height of the Map
MAX_EPSILON = 1
MIN_EPSILON = 0.05
TOTAL_MAP = 13

DQNAgent = DQN(action_space=ACTIONNUM, use_cnn=False, use_capsule=False)
memory = Memory(MEMORY_SIZE)

minerEnv = MinerEnv(HOST, PORT)
minerEnv.start()

train = False

data_index = []
data_step = []
data_score = []
data_energy = []
data_loss = []

mean_step = 0
mean_score = 0
mean_energy = 0
mean_loss = 0
epsilon_decay_fraction = 0.5

f, (ax1, ax2) = plt.subplots(2, 1)

data_map = []


def epsilon_greedy_policy(e_step, decay_steps, initial_epsilon, end_epsilon):
    print("decay_steps ", decay_steps)
    e_step = min(e_step, decay_steps)
    return ((initial_epsilon - end_epsilon) * (1 - e_step / decay_steps)) + end_epsilon


initial_episode = 0
act_epsisode = 20
epsilon = epsilon_greedy_policy(initial_episode, N_EPISODE * epsilon_decay_fraction, MAX_EPSILON, MIN_EPSILON)
DQNAgent.update_epsilon(epsilon)


out_of_map_count = 0

for episode_i in range(initial_episode, N_EPISODE):
    print("act_epsisode ", act_epsisode)
    try:
        mapID = np.random.randint(1, TOTAL_MAP)
        posID_x = np.random.randint(MAP_MAX_X)
        posID_y = np.random.randint(MAP_MAX_Y)

        while str(mapID) + str(posID_x) + str(posID_y) in data_map:
            mapID = np.random.randint(1, TOTAL_MAP)
            posID_x = np.random.randint(MAP_MAX_X)
            posID_y = np.random.randint(MAP_MAX_Y)

        if len(data_map) >= 189 * (TOTAL_MAP - 1):
            data_map = []

        request = ("map" + str(mapID) + "," + str(posID_x) + "," + str(posID_y) + ",50,100")
        minerEnv.send_map_info(request)

        minerEnv.reset()  # Initialize the game environment
        s, s_dead, s_gold, s_gen, _, finish_object, s_player, finish_object_no_gold, _ = minerEnv.get_state()  # Get the state after reseting.

        total_reward = 0  # The amount of rewards for the entire episode
        terminate = False  # The variable indicates that the episode ends
        maxStep = minerEnv.state.mapInfo.maxStep  # Get the maximum number of steps for each episode in training

        for step in range(0, maxStep):
            action = DQNAgent.act(s, s_dead, s_gold, s_gen, s_player, finish_object, act_epsisode, finish_object_no_gold)  # Getting an action from the DQN model from the state (s)
            if finish_object['0'] and finish_object['1'] and finish_object['2'] and finish_object['3'] and finish_object['4'] and finish_object['5']:
                print("map: " + str(mapID))
                print("x and y ", minerEnv.state.x, minerEnv.state.y)
                left_score_swamp = minerEnv.state.mapInfo.get_obstacle_swamp(minerEnv.state.x - 1, minerEnv.state.y)
                right_score_swamp = minerEnv.state.mapInfo.get_obstacle_swamp(minerEnv.state.x + 1, minerEnv.state.y)
                top_score_swamp = minerEnv.state.mapInfo.get_obstacle_swamp(minerEnv.state.x, minerEnv.state.y - 1)
                bottom_score_swamp = minerEnv.state.mapInfo.get_obstacle_swamp(minerEnv.state.x, minerEnv.state.y + 1)
                center_score_swamp = minerEnv.state.mapInfo.get_obstacle_swamp(minerEnv.state.x, minerEnv.state.y)
                print(left_score_swamp, right_score_swamp, top_score_swamp, bottom_score_swamp, center_score_swamp)

            minerEnv.step(str(action[0]))  # Performing the action in order to obtain the new state
            s_next, s_dead_next, s_gold_next, s_gen_next, gold_dis, finish_object_next, s_player_next, finish_object_no_gold_next, score_dis = minerEnv.get_state()  # Getting a new state
            reward = minerEnv.get_reward(gold_dis, score_dis)  # Getting a reward
            terminate = minerEnv.check_terminate()  # Checking the end status of the episode

            # Add this transition to the memory batch
            memory.push(s, s_dead, s_gold, s_gen, s_player, action[0], action[1], reward, terminate, s_next, s_dead_next, s_gold_next, s_gen_next, s_player_next)

            # Sample batch memory to train network
            if memory.length > INITIAL_REPLAY_SIZE:
                batch = memory.sample(BATCH_SIZE)  # Get a BATCH_SIZE experiences for replaying
                loss = DQNAgent.replay(batch, BATCH_SIZE)  # Do relaying
                mean_loss += loss
                train = True  # Indicate the training starts

            total_reward = total_reward + reward  # Plus the reward to the total rewad of the episode
            s = s_next  # Assign the next state for the next step.
            s_dead = s_dead_next
            s_gold = s_gold_next
            s_gen = s_gen_next
            s_player = s_player_next
            finish_object = finish_object_next
            finish_object_no_gold = finish_object_no_gold_next

            # Saving data to file
            save_data = np.hstack(
                [episode_i + 1, step + 1, reward, total_reward, action[0], DQNAgent.epsilon, terminate]).reshape(1, 7)
            with open(filename, 'a') as f:
                pd.DataFrame(save_data).to_csv(f, encoding='utf-8', index=False, header=False)

            if terminate:
                # If the episode ends, then go to the next episode
                break

        act_epsisode = epsilon_greedy_policy(20, N_EPISODE * epsilon_decay_fraction, 20, 10)

        # Iteration to save the network architecture and weights
        if np.mod(episode_i + 1, SAVE_NETWORK) == 0 and train:
            DQNAgent.target_train()  # Replace the learning weights for target model with soft replacement
            # Save the DQN model
            now = datetime.datetime.now()  # Get the latest datetime
            DQNAgent.save_model(str(N_EPISODE) + "_" + str(TOTAL_MAP),
                                "DQNmodel_" + now.strftime("%Y%m%d-%H%M") + "_ep" + str(episode_i + 1))

        # Print the training information after the episode
        print(
            'Episode %d ends. Number of steps is: %d. Accumulated Reward = %.2f. Gold = %.2f. Epsilon = %.2f .Termination code: %d' % (
                episode_i + 1, step + 1, total_reward, minerEnv.state.score, DQNAgent.epsilon, terminate))

        # Decreasing the epsilon if the replay starts
        if train:
            epsilon = epsilon_greedy_policy(episode_i, N_EPISODE * epsilon_decay_fraction, MAX_EPSILON, MIN_EPSILON)
            DQNAgent.update_epsilon(epsilon)

        if episode_i % MEAN_STEP == 0:
            if len(data_index) > MEAN_STEP:
                data_index = data_index[1:]
                data_score = data_score[1:]
                data_energy = data_energy[1:]
                data_step = data_step[1:]
                data_loss = data_loss[1:]

            data_index.append(episode_i / 1000)
            data_score.append(mean_score / MEAN_STEP)
            data_energy.append(mean_energy / MEAN_STEP)
            data_step.append(mean_step / MEAN_STEP)
            data_loss.append(mean_loss / MEAN_STEP)

            mean_step = 0
            mean_score = 0
            mean_energy = 0
            mean_loss = 0

        if episode_i % (2 * MEAN_STEP) == 0:
            plt.savefig('use_dense_2 + ' + str(N_EPISODE) + '_' + str(TOTAL_MAP) + '.png')

        mean_step += step + 1
        if minerEnv.state.score < 0:
            mean_score += 0
        else:
            mean_score += minerEnv.state.score

        mean_energy += minerEnv.state.energy

        if minerEnv.state.status == State.STATUS_ELIMINATED_WENT_OUT_MAP:
            out_of_map_count += 1
        print("STATUS_ELIMINATED_WENT_OUT_MAP ", str(out_of_map_count))

        if minerEnv.state.status == State.STATUS_ELIMINATED_OUT_OF_ENERGY:
            print("STATUS_ELIMINATED_OUT_OF_ENERGY")

        # cyan: step
        # pink: score
        # gray: energy
        ax1.plot(data_index, data_step, label="step", color='cyan')
        ax2.plot(data_index, data_score, label="score", color='pink')

        plt.pause(0.05)

    except Exception as e:
        import traceback

        traceback.print_exc()
        # print("Finished.")
        break