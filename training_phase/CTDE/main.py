import os
import numpy as np
from environment import FireEnvironment
from agent import DDQN_Agent

from PIL import Image as im
import copy

import sys
np.set_printoptions(threshold=sys.maxsize)
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
            rel_pos_j_arr.append((val.x, val.y))

        for val in self.vector.dist_j.values():
            dist_j_arr.append(val)

        for val in self.vector.act_j.values():
            act_j_arr.append(val)

        out_arr = []
        for i in range(len(rel_pos_j_arr)):
            out_arr.append(rel_pos_j_arr[i][0])
            out_arr.append(rel_pos_j_arr[i][1])
            out_arr.append(dist_j_arr[i])
            out_arr.append(act_j_arr[i])

        return np.hstack([self.vector.pos.x, self.vector.pos.y, out_arr])

    def get_Image_obs(self):
        return np.array([self.image.belief_map, self.image.coverage_map])

    def UpdateVectorFromOthers(self, obss):
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
    n_episodes = 2500 # 3000
    n_timestamp = 320 # 320
    learning_rate = 0.0003
    batch_size = 8 # 8
    kick_start_timesteps = 12

    n_timestamp = n_timestamp + kick_start_timesteps

    n_drones = 3 # 3
    n_vector_obs = 2 + (4 * (n_drones-1))
    n_image_obs = (x_len, y_len)
    n_image_channel = 2
    n_actions = 4

    gamma = 0.9
    tau = 0.001
    memory_size = 100000
    update_step = 1

    e1 = 80000 # 80000
    e2 = 800000 # 800000
    e3 = 3200000 # 3200000

    frame_num = 0
    miss_history = []
    reward_history = []
    coverage_history = []
    # loss_history = []
    least_miss = np.full(n_drones, 1.0)

    model_path = os.path.join('models', "t" + str(datetime.now()))

    agent = DDQN_Agent(x_len, y_len, n_drones, n_vector_obs, n_image_obs = n_image_obs, n_image_channel = n_image_channel, n_actions = n_actions, batch_size = batch_size, memory_size = memory_size, 
                 update_step = update_step, learning_rate = learning_rate, gamma = gamma, tau = tau, model_path = model_path)
    
    load_path = "" # "t2023-07-05 18:35:52.268716"      * * * Enter the Folder Name to copy the models and other score histories * * * 

    if load_path == "":
        if not os.path.exists(model_path):
            os.makedirs(model_path)
    else:
        load_path = os.path.join('models', load_path)
        agent.load_models(load_path)
        print(' ... loaded model for all Drones ... ')
        model_path = load_path

        miss_history = np.loadtxt(os.path.join(model_path, "miss_history.csv"), delimiter=",", dtype=np.float32)
        least_miss = miss_history.min(axis=0)
        miss_history = miss_history.tolist()

        reward_history = np.loadtxt(os.path.join(model_path, "reward_history.csv"), delimiter=",", dtype=np.float32)
        reward_history = reward_history.tolist()

        coverage_history = np.loadtxt(os.path.join(model_path, "coverage_history.csv"), delimiter=",", dtype=np.float32)
        coverage_history = coverage_history.tolist()

        # loss_history = np.loadtxt(os.path.join(model_path, "loss_history.csv"), delimiter=",", dtype=np.float32)
        # loss_history = loss_history.tolist()

    actions = ["North", "East", "South", "West"]

    env = FireEnvironment()

    for episode in range(n_episodes):
        # print(datetime.now())

        env.reset()
        obss = [Observation_space((i+1)) for i in range(n_drones)]

        # This is important for the drones to navigate towards the fire.
        for i in range(n_drones):
            obss[i].image.belief_map = env.binary_val.copy()

        # obss[0].vector.pos.x = 60
        # obss[0].vector.pos.y = 40
        # obss[1].vector.pos.x = 40
        # obss[1].vector.pos.y = 60
        # obss[2].vector.pos.x = 40
        # obss[2].vector.pos.y = 40
        # obss[3].vector.pos.x = 60
        # obss[3].vector.pos.y = 60

        for _ in range(kick_start_timesteps):
            env.simStep()
        
        for i in range(n_drones):
            obss[i].UpdateVectorFromOthers(obss)
            _ = env.reward(obss[i]) # Initializing the obs's belief map & coverage map w.r.to currrent state fire map.

        Mt = []
        fire_map_count = 0
        rew_o_ep = []

        fire_map_coverage = np.full((x_len, y_len), 0)

        for ti in range(n_timestamp - kick_start_timesteps):
            if ti % 4 == 0 :
                env.simStep()

            fire_map = env.get_fire_map()
            fire_map_count += np.sum(fire_map)

            frame_num += 1

            if frame_num <= e1:
                epsilon = 1
            elif frame_num <= e2:
                epsilon -= ((1.0 - 0.1) / (e2 - e1))
            elif frame_num <= e3:
                epsilon -= (0.1 / (e3 - e2))
            else:
                epsilon = 0
            
            vector_observations = []
            image_observations = []
            for obs in obss:
                vector_observations.append(obs.get_Vector_obs())
                image_observations.append(obs.get_Image_obs())
            # im.fromarray(np.hstack(obss[i].get_Image_obs()).astype(np.float32) * 255).show()
            action_ids = agent.act(np.array(vector_observations), np.array(image_observations), epsilon = epsilon)
            action_arr = [actions[action_id] for action_id in action_ids]
            for i, action_id in enumerate(action_ids):
                obss[i].action = action_id
            curr_obss_copy = copy.deepcopy(obss)
            new_obss, reward, done, _ = env.step(obss, action_arr)
            for new_obs in new_obss:
                new_obs.UpdateVectorFromOthers(obss)
            
            old_vector_observations, old_image_observations, new_vector_observations, new_image_observations = [], [], [], []
            for i in range(n_drones):
                old_vector_observations.append(curr_obss_copy[i].get_Vector_obs())
                old_image_observations.append(curr_obss_copy[i].get_Image_obs())
                new_vector_observations.append(new_obss[i].get_Vector_obs())
                new_image_observations.append(new_obss[i].get_Image_obs())
            agent.step(np.array(old_vector_observations), np.array(old_image_observations), action_ids, reward, np.array(new_vector_observations), np.array(new_image_observations), done)
            # obss = new_obss # No need obss = new_obss, as the copy is by reference.
            
            rew_o_ep.append(reward)
            
            Mt.append([(np.sum((fire_map - new_obs.image.belief_map).clip(0))) for new_obs in new_obss])
            if (episode % 10 == 0) and (frame_num > e1):
                env.render(obss)
            ConsensusImageObs(obss)

            fire_map_coverage += obss[0].get_Image_obs()[0]

        policy_update_idx = np.argmax(np.array(rew_o_ep).sum(axis=0))
        agent.update_policy(policy_update_idx)

        Mt = (Mt / fire_map_count)
        total_Mt = np.sum(Mt, axis=0)

        # loss_history.append(agent.loss_value)

        print('Episode ', episode, " FireMiss= {:.2f}".format(np.min(total_Mt) * 100), ' % Rew.Ov.Ep.= ', np.array(rew_o_ep).sum(axis=0), ' Coverage= {:.2f}%'.format((np.sum(fire_map_coverage.clip(max = 1))/len(env.universal_fire_set)) * 100))
        miss_history.append(total_Mt)
        reward_history.append(np.array(rew_o_ep).sum(axis=0))
        coverage_history.append(np.sum(fire_map_coverage.clip(max = 1)) / len(env.universal_fire_set))
        
        save_flag = 0
        for i in range(n_drones):
            if least_miss[i] > total_Mt[i]:
                least_miss[i] = total_Mt[i]
                save_flag = 1
                
        if save_flag == 1:
            agent.save_models()
            print(' ... saved checkpoint - ' + str(least_miss[0]) + '... ')
            np.savetxt(os.path.join(model_path, "miss_history.csv"), np.array(miss_history), delimiter=",", fmt='%.4e')
            np.savetxt(os.path.join(model_path, "reward_history.csv"), np.array(reward_history), delimiter=",", fmt='%.4e')
            np.savetxt(os.path.join(model_path, "coverage_history.csv"), np.array(coverage_history), delimiter=",", fmt='%.4e')
            # np.savetxt(os.path.join(model_path, "loss_history.csv"), np.array(loss_history), delimiter=",", fmt='%.4e')
            save_flag = 0

    agent.save_models()
    print(' ... saved final checkpoint for Drone ... ')
    np.savetxt(os.path.join(model_path, "miss_history.csv"), np.array(miss_history), delimiter=",", fmt='%.4e')
    np.savetxt(os.path.join(model_path, "reward_history.csv"), np.array(reward_history), delimiter=",", fmt='%.4e')
    np.savetxt(os.path.join(model_path, "coverage_history.csv"), np.array(coverage_history), delimiter=",", fmt='%.4e')
    # np.savetxt(os.path.join(model_path, "loss_history.csv"), np.array(loss_history), delimiter=",", fmt='%.4e')