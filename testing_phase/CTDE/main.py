import os
import numpy as np
from environment import FireEnvironment
from agent import DDQN_Agent

from PIL import Image as im
import copy

import sys
np.set_printoptions(threshold=sys.maxsize)
import psutil
from datetime import datetime


class UAV_position:
    def __init__ (self, x_len = 100, y_len = 100):
        self.x = np.random.randint(0, x_len, 1)[0]
        self.y = np.random.randint(0, y_len, 1)[0]

class Relative_position:
    def __init__ (self, x, y):
        self.x = x
        self.y = y

class Image_obs:
    def __init__ (self, x_len = 100, y_len = 100):
        self.belief_map = np.full((x_len, y_len), 0)
        self.coverage_map = np.full((x_len, y_len), 0)

class Vector_obs:
    def __init__ (self, x_len = 100, y_len = 100):
        self.pos = UAV_position(x_len, y_len)
        self.rel_pos_j = {}
        self.dist_j = {}
        self.act_j = {}

class Observation_space:
    def __init__ (self, id, x_len = 100, y_len = 100):
        self.id = id
        self.x_len = x_len
        self.y_len = y_len
        self.vector = Vector_obs(self.x_len, self.y_len)
        self.image = Image_obs(self.x_len, self.y_len)
        self.action = 0

    def get_Vector_obs(self):
        rel_pos_j_arr = []
        dist_j_arr = []
        act_j_arr = []

        for val in self.vector.rel_pos_j.values():
            rel_pos_j_arr.append(val.x)
            rel_pos_j_arr.append(val.y)

        for val in self.vector.dist_j.values():
            dist_j_arr.append(val)

        for val in self.vector.act_j.values():
            act_j_arr.append(val)

        return np.hstack([self.vector.pos.x, self.vector.pos.y, rel_pos_j_arr, dist_j_arr, act_j_arr])

    def get_Image_obs(self):
        # ext_x_lim = self.x_len * 2
        # ext_y_lim = self.y_len * 2
        # ext_x_mid = ext_x_lim//2
        # ext_y_mid = ext_y_lim//2

        # main_bm = np.full((ext_x_lim, ext_x_lim), 0).astype(np.float32)
        # main_bm[(ext_x_mid-int(self.vector.pos.x)):(ext_x_mid + (ext_x_mid-int(self.vector.pos.x))), (ext_y_mid-int(self.vector.pos.y)):(ext_y_mid + (ext_y_mid-int(self.vector.pos.y)))] = self.image.belief_map
        
        # # main_bm = im.fromarray(main_bm.astype(np.int8))
        # # main_bm = np.array(fn.resize(main_bm, size=[25]))

        # return main_bm
        return np.array([self.image.belief_map, self.image.coverage_map])

    def UpdateVectorFromOthers(self, obss):
        cm_lim = self.image.coverage_map.shape
        for obs in obss:
            if obs.id is not self.id:
                self.vector.rel_pos_j[obs.id] = Relative_position((obs.vector.pos.x - self.vector.pos.x),(obs.vector.pos.y - self.vector.pos.y))
                self.vector.dist_j[obs.id] = np.sqrt( (obs.vector.pos.x - self.vector.pos.x)**2 + (obs.vector.pos.y - self.vector.pos.y)**2 )
                self.vector.act_j[obs.id] = obs.action

def ConsensusImageObs(obss):
    dom_bmap = obss[0].image.belief_map.copy()
    dom_cmap = obss[0].image.coverage_map.copy()
    cm_lim = dom_cmap.shape

    for obs in obss[1:]:
        for i in range(cm_lim[0]):
            for j in range(cm_lim[1]):
                if dom_cmap[i][j] < obs.image.coverage_map[i][j]:
                    dom_cmap[i][j] = obs.image.coverage_map[i][j]
                    dom_bmap[i][j] = obs.image.belief_map[i][j]

    for obs in obss:
        obs.image.coverage_map = dom_cmap.copy()
        obs.image.belief_map = dom_bmap.copy()


if __name__ == '__main__':

    x_len = 100
    y_len = 100
    n_episodes = 1 # 2000
    n_timestamp = 320 # 320
    kick_start_timesteps = 12

    n_timestamp = n_timestamp + kick_start_timesteps

    n_drones = 3 # 3
    n_vector_obs = 2 + (4 * (n_drones-1))
    n_image_obs = (x_len, y_len)
    n_image_channel = 2
    n_actions = 4

    coverage = []
    rew_o_ep = []
    Mt = []

    model_path = os.path.join('models', "t" + str(datetime.now()))

    agents = [DDQN_Agent(x_len, y_len, n_vector_obs, n_image_obs = n_image_obs, n_image_channel = n_image_channel, n_actions = n_actions,
                 model_path = model_path, id = i) for i in range(n_drones)]
    
    load_path = "t2023-08-17 16:43:57.654315" # "t2023-07-05 18:35:52.268716"      * * * Enter the Folder Name to copy the models and other score histories * * * 


    load_path = os.path.join('models', load_path)
    for id in range(n_drones):
        agents[id].load_models(load_path, id)
    print(' ... loaded model for all Drones ... ')
    model_path = load_path

    actions = ["North", "East", "South", "West"]

    env = FireEnvironment()

    for episode in range(n_episodes):
        # Reset Env and Observation_space
        env.reset()
        obs = [Observation_space((i+1)) for i in range(n_drones)]

        # This is important for the drones to navigate towards the fire.
        for i in range(n_drones):
            obs[i].image.belief_map = env.binary_val.copy()

        for _ in range(kick_start_timesteps):
            env.simStep()
        
        for i in range(n_drones):
            obs[i].UpdateVectorFromOthers(obs)
            _ = env.reward(obs[i]) # Initializing the obs's belief map & coverage map from fire map w.r.to currrent obs.

        fire_map_count = 0
        
        fire_map = env.get_fire_map()
        # fire_map_count += np.sum(fire_map)

        fire_map_coverage = np.full((x_len, y_len), 0)

        for ti in range(n_timestamp - kick_start_timesteps):
            if ti % 4 == 0 :
                env.simStep()

            Mi_all_drones = []
            rew_o_ts = []

            coverage_per_ts = np.full((x_len, y_len), 0)
            fire_map = env.get_fire_map()
            fire_map_count += np.sum(fire_map)

            for i in range(n_drones):
                
                action_id = agents[i].act(obs[i].get_Vector_obs(), obs[i].get_Image_obs())
                action = actions[action_id]
                obs[i].action = action_id
                curr_obs_copy = copy.deepcopy(obs[i])
                new_obs, reward, done, _ = env.step(obs[i], action)
                new_obs.UpdateVectorFromOthers(obs)

                curr_score = (np.sum((fire_map - new_obs.image.belief_map).clip(0)))
                Mi_all_drones.append(curr_score)
                rew_o_ts.append(reward)
                fire_map_coverage += obs[i].get_Image_obs()[0]
            
            rew_o_ep.append(rew_o_ts)
            
            Mt.append(Mi_all_drones)
            env.render(obs)
            ConsensusImageObs(obs)
            
            
            coverage.append(np.sum(fire_map_coverage.clip(max = 1)) / len(env.universal_fire_set))

        print(" Done...")
        np.savetxt(os.path.join(model_path, "miss_history.csv"), np.array(Mt)/fire_map_count, delimiter=",", fmt='%.4e')
        np.savetxt(os.path.join(model_path, "reward_history.csv"), np.array(rew_o_ep), delimiter=",", fmt='%.4e')
        np.savetxt(os.path.join(model_path, "coverage_history.csv"), np.array(coverage), delimiter=",", fmt='%.4e')

