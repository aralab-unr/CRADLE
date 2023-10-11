import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from model import DDQN_Graph

from datetime import datetime
# from PIL import Image as im

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
        self.belief_map = np.full((x_len, y_len), 0, dtype=np.int8)
        self.coverage_map = np.full((x_len, y_len), 0, dtype=np.uint8)

class Vector_obs:
    def __init__ (self, x_len = 100, y_len = 100, n_drones = 3):
        self.pos = UAV_position(x_len, y_len)
        self.rel_pos_j = [ Relative_position(-200, -200) for _ in range(n_drones - 1)]
        self.dist_j = [ -1 for _ in range(n_drones - 1)]
        self.act_j = [ -1 for _ in range(n_drones - 1)]

class Observation_space:
    def __init__ (self, x_len = 100, y_len = 100, n_drones = 3):
        self.x_len = x_len
        self.y_len = y_len
        self.n_drones = n_drones
        self.vector = Vector_obs(self.x_len, self.y_len, self.n_drones)
        self.image = Image_obs(self.x_len, self.y_len)
        self.action = -1

    def get_Vector_obs(self):
        rel_pos_j_arr = []
        dist_j_arr = []
        act_j_arr = []

        for val in self.vector.rel_pos_j:
            rel_pos_j_arr.append((val.x, val.y))

        for val in self.vector.dist_j:
            dist_j_arr.append(val)

        for val in self.vector.act_j:
            act_j_arr.append(val)

        out_arr = []
        for i in range(len(rel_pos_j_arr)):
            out_arr.append(rel_pos_j_arr[i][0])
            out_arr.append(rel_pos_j_arr[i][1])
            out_arr.append(dist_j_arr[i])
            out_arr.append(act_j_arr[i])

        return np.hstack([self.vector.pos.x, self.vector.pos.y, out_arr]).astype(np.float32)

    def get_Image_obs(self):
        return np.array([self.image.belief_map, self.image.coverage_map], dtype=np.uint8)
    
    def share_position(self):
        return (self.vector.pos.x, self.vector.pos.y)
    
    def share_obs(self):
        return (self.vector.pos.x, self.vector.pos.y, self.image.belief_map, self.image.coverage_map, self.action)
    
    def reset(self):
        self.vector = Vector_obs(self.x_len, self.y_len, self.n_drones)
        self.image = Image_obs(self.x_len, self.y_len)

class DDQN_Agent(): 
    """docstring for ddqn_agent"""
    def __init__(self, x_len, y_len, n_vector_obs, n_image_obs, n_image_channel, n_actions, model_path, id):
        super(DDQN_Agent, self).__init__()
        # state space dimension
        self.x_len = x_len
        self.y_len = y_len
        self.n_vector_obs = n_vector_obs
        self.n_image_obs = n_image_obs
        self.n_image_channel = n_image_channel
        # action space dimension
        self.n_actions = n_actions
        # configuration
        self.model_path = model_path
        self.id = id

        # check cpu or gpu
        self.setup_gpu()
        # initialize model graph
        self.setup_model()
    
    def setup_gpu(self): 
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def setup_model(self):
        self.policy_model = DDQN_Graph(
            self.n_vector_obs,
            self.n_image_channel, 
            self.n_actions,
            self.model_path,
            'policy_model_' + str(self.id)).to(self.device)

    def prepare_map(self, map_in, x, y):
        ext_x_lim = self.x_len * 2
        ext_y_lim = self.y_len * 2
        ext_x_mid = ext_x_lim//2 
        ext_y_mid = ext_y_lim//2

        mod_map = np.full((self.n_image_channel, ext_x_lim, ext_y_lim), 0)
        for idx in range(self.n_image_channel):
            mod_map[idx, (ext_x_mid-int(x)):(ext_x_mid + (ext_x_mid-int(x))), (ext_y_mid-int(y)):(ext_y_mid + (ext_y_mid-int(y)))] = map_in[idx]
        
        # main_bm = im.fromarray(main_bm.astype(np.int8))
        # main_bm = np.array(fn.resize(main_bm, size=[25]))

        return mod_map
    
    def act(self, vector_obs, mod_map):
        mod_map_obs = self.prepare_map(mod_map, vector_obs[0], vector_obs[1])
        # take an action for a time step
        # state: 1, state_size
        vector_obs = torch.tensor(np.array([vector_obs])).float().to(self.device)
        mod_map_obs = torch.tensor(np.array([mod_map_obs])).float().to(self.device)
        # inference by policy model
        self.policy_model.eval()
        with torch.no_grad(): 
            # action_vs: 1, action_size
            action_vs = self.policy_model(vector_obs, mod_map_obs)
        self.policy_model.train()
        return action_vs.cpu().detach().numpy()
    
    def load_models(self, load_path, id):
        self.model_path = load_path
        self.policy_model.load_checkpoint(load_path, "policy_model_" + str(id), self.device)

    def adapt_input_layer(self, n_vector_obs):
        self.policy_model.dense_layer_1 = nn.Linear(n_vector_obs, 50)
        self.policy_model.eval()

        nn.init.kaiming_normal_(self.policy_model.dense_layer_1.weight, nonlinearity='relu')
        nn.init.zeros_(self.policy_model.dense_layer_1.bias)
        
