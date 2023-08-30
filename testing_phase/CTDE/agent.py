import torch
import torch.nn.functional as F

import numpy as np

from model import DDQN_Graph


class DDQN_Agent(): 
    """docstring for ddqn_agent"""
    def __init__(self, x_len, y_len, n_vector_obs, n_image_obs, n_image_channel, n_actions, 
                 model_path, id):
        super(DDQN_Agent, self).__init__()
        # state space dimension
        self.x_len = x_len
        self.y_len = y_len
        self.n_vector_obs = n_vector_obs
        self.n_image_obs = n_image_obs
        self.n_image_channel = n_image_channel
        # action space dimension
        self.n_actions = n_actions
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

        mod_map = np.full((self.n_image_channel, ext_x_lim, ext_x_lim), 0).astype(np.float32)
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
        return np.argmax(action_vs.cpu().detach().numpy())

    def load_models(self, load_path, id):
        self.model_path = load_path
        self.policy_model.load_checkpoint(load_path, "policy_model_" + str(id))
