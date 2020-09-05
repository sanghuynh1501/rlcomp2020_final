import math
from MINER_STATE import State
from DQNModel_Torch_Bot import DQN
import numpy as np

GroundID = 0
TreeID = 1
TrapID = 2
SwampID = 3

MAP_MAX_X = 20  # Width of the Map
MAP_MAX_Y = 8  # Height of the Map
ACTIONNUM = 5  # The number of actions output from the DQN model
EPSILON = 0.05


class PlayerInfo:
    def __init__(self, id):
        self.playerId = id
        self.score = 0
        self.energy = 0
        self.posx = 0
        self.posy = 0
        self.lastAction = -1
        self.status = 0
        self.freeCount = 0


class Bot2:
    ACTION_GO_LEFT = 0
    ACTION_GO_RIGHT = 1
    ACTION_GO_UP = 2
    ACTION_GO_DOWN = 3
    ACTION_FREE = 4
    ACTION_CRAFT = 5

    def __init__(self, id):
        self.state = State()
        self.info = PlayerInfo(id)
        self.network = DQN(action_space=ACTIONNUM, epsilon=EPSILON)

    def position_encode(self, shape, position):
        result = np.zeros(shape)
        for i in range(shape[0]):
            one_hot = np.zeros((shape[1],))
            if position[i] == 1:
                one_hot[i] = 1
            result[i, :] = one_hot
        return result

    def one_hot_encoder(self, input_tensor, is_gold, is_player):
        if input_tensor == -1:
            if is_gold:
                input_tensor = 4
            elif is_player:
                input_tensor = 0
            else:
                input_tensor = 5
        output = np.zeros((6,), dtype=np.float32).tolist()
        output[input_tensor] = 1
        return output

    def distance_caculate(self, x1, x2, y1, y2):
        x = math.pow((x1 - x2), 2)
        y = math.pow((y1 - y2), 2)
        return math.sqrt(x + y)

    def find_gold_after(self, posx, posy, player_object, type_str, total_player):
        gold_near = []
        gold_near_result = []
        is_gold_near = False
        player_count = 0

        while not is_gold_near and player_count <= total_player:
            spy_x = posx
            spy_y = posy

            if type_str == "left":
                spy_x -= 1
            if type_str == "right":
                spy_x += 1
            if type_str == "top":
                spy_y -= 1
            if type_str == "bottom":
                spy_y += 1

            while 0 <= spy_x <= MAP_MAX_X and 0 <= spy_y <= MAP_MAX_Y:
                have_gold = False
                if type_str == "left" or type_str == 'right':
                    have_gold = self.state.mapInfo.gold_amount_x(spy_x, player_object, player_count)
                if type_str == "top" or type_str == "bottom":
                    have_gold = self.state.mapInfo.gold_amount_y(spy_y, player_object, player_count)

                if have_gold > 0:
                    gold_near = [spy_x, spy_y]
                    break

                if type_str == "left":
                    spy_x -= 1
                if type_str == "right":
                    spy_x += 1
                if type_str == "top":
                    spy_y -= 1
                if type_str == "bottom":
                    spy_y += 1

            if type_str == "left":
                if spy_x >= 0:
                    gold_near_result = gold_near
                    is_gold_near = True
            if type_str == "right":
                if spy_x <= MAP_MAX_X:
                    gold_near_result = gold_near
                    is_gold_near = True
            if type_str == "top":
                if spy_y >= 0:
                    gold_near_result = gold_near
                    is_gold_near = True
            if type_str == "bottom":
                if spy_y <= MAP_MAX_Y:
                    gold_near_result = gold_near
                    is_gold_near = True

            player_count += 1

        return gold_near_result, is_gold_near

    def find_gold(self, posx, posy, player_object, type_str, total_player, gold_object):

        gold_near = []
        gold_near_result = []
        is_gold_near = False
        player_count = 0
        while not is_gold_near and player_count <= total_player:

            spy_x = posx
            spy_y = posy

            if type_str == "left":
                spy_x -= 1
            if type_str == "right":
                spy_x += 1
            if type_str == "top":
                spy_y -= 1
            if type_str == "bottom":
                spy_y += 1

            while 0 <= spy_x <= MAP_MAX_X and 0 <= spy_y <= MAP_MAX_Y:
                if player_count == 0:
                    have_player = str(spy_x) + '_' + str(spy_y) not in player_object
                else:
                    if str(spy_x) + '_' + str(spy_y) in player_object:
                        have_player = player_object[str(spy_x) + '_' + str(spy_y)] == player_count
                    else:
                        have_player = False

                have_gold = self.check_gold_object(spy_x, spy_y, gold_object) > 0 and have_player
                if have_gold:
                    gold_near = [spy_x, spy_y]
                    break

                if type_str == "left":
                    spy_x -= 1
                if type_str == "right":
                    spy_x += 1
                if type_str == "top":
                    spy_y -= 1
                if type_str == "bottom":
                    spy_y += 1

            if type_str == "left":
                if spy_x >= 0:
                    gold_near_result = gold_near
                    is_gold_near = True
            if type_str == "right":
                if spy_x <= MAP_MAX_X:
                    gold_near_result = gold_near
                    is_gold_near = True
            if type_str == "top":
                if spy_y >= 0:
                    gold_near_result = gold_near
                    is_gold_near = True
            if type_str == "bottom":
                if spy_y <= MAP_MAX_Y:
                    gold_near_result = gold_near
                    is_gold_near = True

            player_count += 1

        return gold_near_result, is_gold_near

    def log_output(self, pos):
        return np.log10(pos + 2)

    def get_position_gold(self, posx, posy, player_object, center_gold, total_player, gold_object):

        if center_gold:
            return [posx, posy], -1, 0

        left_pos, left_gold = self.find_gold(posx, posy, player_object, "left", total_player, gold_object)
        right_pos, right_gold = self.find_gold(posx, posy, player_object, "right", total_player, gold_object)
        top_pos, top_gold = self.find_gold(posx, posy, player_object, "top", total_player, gold_object)
        bottom_pos, bottom_gold = self.find_gold(posx, posy, player_object, "bottom", total_player, gold_object)

        gold_nears = {
            "0": left_pos,
            "1": right_pos,
            "2": top_pos,
            "3": bottom_pos,
        }
        gold_nears_boolean = {
            "0": left_gold,
            "1": right_gold,
            "2": top_gold,
            "3": bottom_gold,
        }

        min_distance = 0.
        gold_distance = 0.
        min_gold = []
        index = -1

        for i in range(0, 4):
            if gold_nears_boolean[str(i)]:
                player_count = 1
                if str(gold_nears[str(i)][0]) + "_" + str(gold_nears[str(i)][1]) in player_object:
                    player_count = player_object[str(gold_nears[str(i)][0]) + "_" + str(gold_nears[str(i)][1])] + 1
                distance = (self.check_gold_object(gold_nears[str(i)][0], gold_nears[str(i)][1],
                                                   gold_object) / player_count) / self.distance_caculate(posx,
                                                                                                         gold_nears[
                                                                                                             str(i)][0],
                                                                                                         posy,
                                                                                                         gold_nears[
                                                                                                             str(i)][1])
                if distance > min_distance:
                    min_gold = gold_nears[str(i)]
                    gold_amout = self.check_gold_object(gold_nears[str(i)][0], gold_nears[str(i)][1],
                                                        gold_object) / player_count
                    min_distance = distance
                    gold_distance = self.distance_caculate(posx, gold_nears[str(i)][0], posy, gold_nears[str(i)][1])
                    index = i
                else:
                    if distance == min_distance:
                        if self.check_gold_object(gold_nears[str(i)][0], gold_nears[str(i)][1],
                                                  gold_object) / player_count > gold_amout:
                            min_gold = gold_nears[str(i)]
                            gold_amout = self.check_gold_object(gold_nears[str(i)][0], gold_nears[str(i)][1],
                                                                gold_object) / player_count
                            min_distance = distance
                            gold_distance = self.distance_caculate(posx, gold_nears[str(i)][0], posy,
                                                                   gold_nears[str(i)][1])
                            index = i

        if not left_gold and not right_gold and not top_gold and not bottom_gold:
            left_pos, left_gold = self.find_gold_after(posx, posy, player_object, "left", total_player)
            right_pos, right_gold = self.find_gold_after(posx, posy, player_object, "right", total_player)
            top_pos, top_gold = self.find_gold_after(posx, posy, player_object, "top", total_player)
            bottom_pos, bottom_gold = self.find_gold_after(posx, posy, player_object, "bottom", total_player)
            gold_nears = {
                "0": left_pos,
                "1": right_pos,
                "2": top_pos,
                "3": bottom_pos,
            }
            gold_nears_boolean = {
                "0": left_gold,
                "1": right_gold,
                "2": top_gold,
                "3": bottom_gold,
            }

            min_distance = 0.
            gold_distance = 0.
            gold_amout = 0.
            min_gold = []
            index = -1

            for i in range(0, 4):
                if gold_nears_boolean[str(i)]:
                    gold_pos = self.state.mapInfo.get_gold_pos([gold_nears[str(i)][0], gold_nears[str(i)][1]])
                    for pos in gold_pos:
                        player_count = 1
                        if str(pos[0]) + "_" + str(pos[1]) in player_object:
                            player_count = player_object[str(pos[0]) + "_" + str(pos[1])] + 1
                        distance = (self.check_gold_object(pos[0], pos[1],
                                                           gold_object) / player_count) / self.distance_caculate(posx,
                                                                                                                 pos[0],
                                                                                                                 posy,
                                                                                                                 pos[1])
                        if distance > min_distance:
                            min_gold = pos
                            gold_amout = self.check_gold_object(pos[0], pos[1], gold_object) / player_count
                            min_distance = distance
                            gold_distance = self.distance_caculate(posx, pos[0], posy, pos[1])
                            index = i
                        else:
                            if distance == min_distance:
                                if self.check_gold_object(pos[0], pos[1], gold_object) / player_count > gold_amout:
                                    min_gold = pos
                                    gold_amout = self.check_gold_object(pos[0], pos[1], gold_object) / player_count
                                    min_distance = distance
                                    gold_distance = self.distance_caculate(posx, pos[0], posy, pos[1])
                                    index = i

        return min_gold, index, gold_distance

    def check_swamp(self, x, y, swamp_object):
        if str(x) + "_" + str(y) in swamp_object:
            return swamp_object[str(x) + "_" + str(y)]
        else:
            return 0

    def check_obstacle_object(self, x, y, obstacle_object):
        if str(x) + "_" + str(y) in obstacle_object:
            return obstacle_object[str(x) + "_" + str(y)]
        else:
            return -1

    def check_gold_object(self, x, y, gold_object):
        if str(x) + "_" + str(y) in gold_object:
            return gold_object[str(x) + "_" + str(y)]
        else:
            return 0

    # Functions are customized by client
    def get_state(self):
        player_object = {}
        player_object_zero = {}
        swamp_object, obstacle_object = self.state.mapInfo.get_obstacle_swamp_object()
        gold_object = self.state.mapInfo.get_gold_amount_object()

        total_player = 0
        for player in self.state.players:
            if player["playerId"] != self.state.id:
                total_player += 1

        for cell in self.state.mapInfo.golds:
            for player in self.state.players:
                if player["playerId"] != self.state.id:
                    player_dis = self.distance_caculate(cell["posx"], player["posx"], cell["posy"], player["posy"])
                    agent_dis = self.distance_caculate(cell["posx"], self.state.x, cell["posy"], self.state.y)
                    if player_dis <= agent_dis:
                        position_string = str(cell["posx"]) + "_" + str(cell["posy"])
                        if position_string in player_object:
                            player_object[position_string] += 1
                        else:
                            player_object[position_string] = 1

        for cell in self.state.mapInfo.golds:
            for player in self.state.players:
                if player["playerId"] != self.state.id:
                    player_dis = self.distance_caculate(cell["posx"], player["posx"], cell["posy"], player["posy"])
                    if player_dis == 0:
                        position_string = str(cell["posx"]) + "_" + str(cell["posy"])
                        if position_string in player_object_zero:
                            player_object_zero[position_string] += 1
                        else:
                            player_object_zero[position_string] = 1

        state_position = np.zeros((5, 5, 6))

        posx = self.state.x
        posy = self.state.y
        state_top = [posx - 2, posy - 2]

        finish_object = {
            "0": False,
            "1": False,
            "2": False,
            "3": False,
            "4": False,
            "5": False,
            "6": False
        }

        dead_object = {
            "0": False,
            "1": False,
            "2": False,
            "3": False,
            "4": False,
            "5": False
        }

        left_score_swamp = self.check_swamp(posx - 1, posy, swamp_object) - self.state.energy >= 0
        right_score_swamp = self.check_swamp(posx + 1, posy, swamp_object) - self.state.energy >= 0
        top_score_swamp = self.check_swamp(posx, posy - 1, swamp_object) - self.state.energy >= 0
        bottom_score_swamp = self.check_swamp(posx, posy + 1, swamp_object) - self.state.energy >= 0
        center_score_swamp = self.check_swamp(posx, posy, swamp_object) - self.state.energy >= 0

        left_score_trap = self.check_obstacle_object(posx - 1, posy,
                                                     obstacle_object) == TrapID and self.state.energy <= 10
        right_score_trap = self.check_obstacle_object(posx + 1, posy,
                                                      obstacle_object) == TrapID and self.state.energy <= 10
        top_score_trap = self.check_obstacle_object(posx, posy - 1,
                                                    obstacle_object) == TrapID and self.state.energy <= 10
        bottom_score_trap = self.check_obstacle_object(posx, posy + 1,
                                                       obstacle_object) == TrapID and self.state.energy <= 10
        center_score_trap = self.check_obstacle_object(posx, posy,
                                                       obstacle_object) == TrapID and self.state.energy <= 10

        left_tree = self.check_obstacle_object(posx - 1, posy, obstacle_object) == TreeID and self.state.energy <= 20
        right_tree = self.check_obstacle_object(posx + 1, posy, obstacle_object) == TreeID and self.state.energy <= 20
        top_tree = self.check_obstacle_object(posx, posy - 1, obstacle_object) == TreeID and self.state.energy <= 20
        bottom_tree = self.check_obstacle_object(posx, posy + 1, obstacle_object) == TreeID and self.state.energy <= 20
        center_tree = self.check_obstacle_object(posx, posy, obstacle_object) == TreeID and self.state.energy <= 20

        center_gold = self.check_gold_object(posx, posy, gold_object) > 0

        ground = self.check_gold_object(posx, posy, gold_object) == 0
        no_ground = self.state.energy <= 5

        if posx == 0 and posy == 0:
            dead_object["0"] = True
            dead_object["2"] = True
        elif posx == 0 and posy == MAP_MAX_Y:
            dead_object["0"] = True
            dead_object["3"] = True
        elif posx == MAP_MAX_X and posy == 0:
            dead_object["1"] = True
            dead_object["2"] = True
        elif posx == MAP_MAX_X and posy == MAP_MAX_Y:
            dead_object["1"] = True
            dead_object["3"] = True
        elif posx == 0:
            dead_object["0"] = True
        elif posx == MAP_MAX_X:
            dead_object["1"] = True
        elif posy == 0:
            dead_object["2"] = True
        elif posy == MAP_MAX_Y:
            dead_object["3"] = True

        min_gold, index, gold_distance = self.get_position_gold(posx, posy, player_object_zero, center_gold,
                                                                total_player, gold_object)
        gold_only_left = index == 0
        gold_only_right = index == 1
        gold_only_top = index == 2
        gold_only_bottom = index == 3

        gold_position_encode = np.array([
            int(min_gold[0] < posx),  # food left
            int(min_gold[0] > posx),  # food right
            int(min_gold[1] < posy),  # food up
            int(min_gold[1] > posy)  # food down
        ])

        player_distances = []
        player_ids = []
        for player in self.state.players:
            if player["playerId"] != self.state.id:
                player_dis = self.distance_caculate(min_gold[0], player["posx"], min_gold[1], player["posy"])
                player_distances.append(player_dis)
                player_ids.append(player["playerId"])

        agent_dis = self.distance_caculate(min_gold[0], self.state.x, min_gold[1], self.state.y)
        player_distances.append(agent_dis)
        player_ids.append(self.state.id)

        player_ids = [x for _, x in sorted(zip(player_distances, player_ids))]
        score_dis = player_ids.index(self.state.id) * 20

        player_count_encode = np.array([])
        for player in self.state.players:
            if player["playerId"] != self.state.id:
                player_pos = self.log_output(np.array(
                    [(player["posx"] - self.state.x) / MAP_MAX_X, (player["posy"] - self.state.y) / MAP_MAX_Y]))
                if len(player_count_encode) == 0:
                    player_count_encode = player_pos
                else:
                    player_count_encode = np.concatenate([player_count_encode, player_pos])
        for player in self.state.players:
            if player["playerId"] != self.state.id:
                player_pos = self.log_output(
                    np.array([(player["posx"] - min_gold[0]) / MAP_MAX_X, (player["posy"] - min_gold[1]) / MAP_MAX_Y]))
                player_count_encode = np.concatenate([player_count_encode, player_pos])

        gold_position_encode = np.where(gold_position_encode > 0.5, 0.9, gold_position_encode)
        gold_position_encode = np.where(gold_position_encode < 0.5, 0.1, gold_position_encode)

        finish_object["0"] = left_score_swamp or left_score_trap or left_tree or dead_object[
            "0"] or center_gold or gold_only_right or gold_only_top or gold_only_bottom
        finish_object["1"] = right_score_swamp or right_score_trap or right_tree or dead_object[
            "1"] or center_gold or gold_only_left or gold_only_top or gold_only_bottom
        finish_object["2"] = top_score_swamp or top_score_trap or top_tree or dead_object[
            "2"] or center_gold or gold_only_left or gold_only_right or gold_only_bottom
        finish_object["3"] = bottom_score_swamp or bottom_score_trap or bottom_tree or dead_object[
            "3"] or center_gold or gold_only_left or gold_only_right or gold_only_top
        finish_object["4"] = self.state.energy >= 50
        finish_object["5"] = ground or no_ground
        finish_object["6"] = center_score_swamp or center_score_trap or center_tree

        finish_object_no_gold = {
            "0": left_score_swamp or left_score_trap or left_tree or dead_object["0"],
            "1": right_score_swamp or right_score_trap or right_tree or dead_object["1"],
            "2": top_score_swamp or top_score_trap or top_tree or dead_object["2"],
            "3": bottom_score_swamp or bottom_score_trap or bottom_tree or dead_object["3"]
        }

        dead_position_encode = np.array([
            int(finish_object["0"]),  # food left
            int(finish_object["1"]),  # food right
            int(finish_object["6"]),
            int(finish_object["2"]),  # food up
            int(finish_object["3"]),  # food down
            int(self.state.energy <= 10)
        ])

        dead_position_encode = np.where(dead_position_encode > 0.5, 0.9, dead_position_encode)
        dead_position_encode = np.where(dead_position_encode < 0.5, 0.1, dead_position_encode)

        for i in range(0, 5):
            state_top[1] = posy - 2
            for j in range(0, 5):
                state_position[i, j, :] = self.one_hot_encoder(
                    self.state.mapInfo.get_obstacle(state_top[0], state_top[1]),
                    self.state.mapInfo.gold_amount(state_top[0], state_top[1]) > 0,
                    state_top[0] == posx and state_top[1] == posy
                )
                state_top[1] += 1
            state_top[0] += 1

        state_engergy = np.ones((100,), dtype=np.float32) * self.state.energy / 50

        return \
            np.expand_dims(np.reshape(state_position, (25, 6)), 0), \
            np.expand_dims(dead_position_encode, 0), \
            np.expand_dims(gold_position_encode, 0), \
            np.expand_dims(state_engergy, 0), gold_distance, finish_object, np.expand_dims(player_count_encode,
                                                                                           0), finish_object_no_gold, score_dis

    def next_action(self):
        s, s_dead, s_gold, s_gen, _, finish_object, s_player, finish_object_no_gold, _ = self.get_state()
        action = self.network.act(s, s_dead, s_gold, s_gen, s_player, finish_object, 0, finish_object_no_gold)
        return int(action[0])

    def new_game(self, data):
        try:
            self.state.init_state(data)
        except Exception as e:
            import traceback
            traceback.print_exc()

    def new_state(self, data):
        # action = self.next_action();
        # self.socket.send(action)
        if self.state.status != State.STATUS_PLAYING:
            print('bot 1 gold ', self.info.score)
        try:
            self.state.update_state(data)
        except Exception as e:
            import traceback
            traceback.print_exc()