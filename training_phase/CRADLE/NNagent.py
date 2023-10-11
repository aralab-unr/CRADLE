import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from model import DDQN_Graph, ReplayMemory

from datetime import datetime
# from PIL import Image as im

class Aggregate_Q(nn.Module):
    def __init__(self):
        super(Aggregate_Q, self).__init__()

    def aggregate(self, q_list):
        # Sum up the Q-values of all agents
        return q_list.mean(dim=0)
    
class LocalizedVDN(nn.Module):
    def __init__(self, w):
        super(LocalizedVDN, self).__init__()
        self.w = w

    def forward(self, Q_vs, Aggd_Nei_Q):
        Q_vs = Q_vs * self.w
        Aggd_Nei_Q = Aggd_Nei_Q * (1 - self.w)
        return torch.stack([Q_vs, Aggd_Nei_Q]).sum(dim=0)
    
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
    def __init__(self, x_len, y_len, n_vector_obs, n_image_obs, n_image_channel, n_actions, batch_size, memory_size, 
                 update_step, learning_rate, gamma, tau, weighted_sum, model_path, id):
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
        self.batch_size = batch_size
        self.update_step = update_step
        self.lr = learning_rate
        self.gamma = gamma
        self.tau = tau
        self.model_path = model_path
        self.id = id
        self.aggregate_Q = Aggregate_Q()
        self.localVDN = LocalizedVDN( w = weighted_sum)

        # check cpu or gpu
        self.setup_gpu()
        # initialize model graph
        self.setup_model()
        # initialize optimizer
        self.setup_opt()
        # enable Replay Memory
        self.memory = ReplayMemory(memory_size, self.n_vector_obs, self.n_image_channel, self.n_image_obs)
    
    def setup_gpu(self): 
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def setup_model(self):
        self.policy_model = DDQN_Graph(
            self.n_vector_obs,
            self.n_image_channel, 
            self.n_actions,
            self.model_path,
            'policy_model_' + str(self.id)).to(self.device)
        self.target_model = DDQN_Graph(
            self.n_vector_obs,
            self.n_image_channel, 
            self.n_actions,
            self.model_path,
            'target_model_' + str(self.id)).to(self.device)
    
    def setup_opt(self):
        self.opt = torch.optim.Adam(self.policy_model.parameters(), lr=self.lr)

    def prepare_map(self, map_in, x, y):
        ext_x_lim = self.x_len * 2
        ext_y_lim = self.y_len * 2
        ext_x_mid = ext_x_lim//2 
        ext_y_mid = ext_y_lim//2

        mod_map = np.full((self.n_image_channel, ext_x_lim, ext_x_lim), 0)
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
    
    def step(self, cur_vec_state, cur_img_state, action, reward, next_vec_state, next_img_state):
        self.memory.push(cur_vec_state, cur_img_state, action, reward, next_vec_state, next_img_state)
    
    def check_train(self, frame_num):
        if (frame_num % self.update_step == 0) and self.get_mem_cntr() >= self.batch_size:
            return True
        return False

    def learn(self, batch_idx, Nei_Q, soft_copy=True):
        # print("Back-propagation starts at ",datetime.now())
        # states: batch_size, state_size
        # actions: batch_size, 1
        # rewards: batch_size, 1
        # next_states: batch_size, state_size
        vec_states, img_states, actions, rewards, next_vec_states, next_img_states = self.memory.sampleByIndex(batch_idx)
        mod_img_states = np.full((self.batch_size, self.n_image_channel, self.x_len * 2, self.y_len * 2), 0)
        mod_next_img_states = np.full((self.batch_size, self.n_image_channel, self.x_len * 2, self.y_len * 2), 0)
        # Return list
        return_lst = []
        
        for i in range(self.batch_size):
            mod_img_states[i] = self.prepare_map(img_states[i], vec_states[i][0], vec_states[i][1])
            mod_next_img_states[i] = self.prepare_map(next_img_states[i], next_vec_states[i][0], next_vec_states[i][1])

        # print("\nlearn: ")
        # print(vec_states)
        # img = im.fromarray(np.hstack(img_states).astype(np.float32) * 255)
        # img.show()
        # print(actions)
        # print(rewards)
        # print(next_vec_states)
        # img = im.fromarray(np.hstack(next_img_states).astype(np.float32) * 255)
        # img.show()

        vec_states = torch.tensor(vec_states).float().to(self.device)
        img_states = torch.tensor(mod_img_states).float().to(self.device)
        actions = torch.tensor(actions).long().to(self.device)
        rewards = torch.tensor(rewards).reshape(-1, 1).float().to(self.device)
        next_vec_states = torch.tensor(next_vec_states).float().to(self.device)
        next_img_states = torch.tensor(mod_next_img_states).float().to(self.device)

        _, next_idx = self.policy_model(next_vec_states, next_img_states).max(1)
        target_next_action_vs = self.target_model(next_vec_states, next_img_states).gather(1, next_idx.unsqueeze(1))
        target_q_vs = rewards + (self.gamma * target_next_action_vs)
        policy_q_vs = self.policy_model(vec_states, img_states).gather(1, actions.unsqueeze(1))
        return_lst.append(torch.mean(policy_q_vs))
        
        if len(Nei_Q[0]) > 0:
            Aggd_Nei_policy_Q = self.aggregate_Q.aggregate(torch.stack(Nei_Q[0]))
            Aggd_Nei_target_Q = self.aggregate_Q.aggregate(torch.stack(Nei_Q[1]))

            Localized_Nei_policy_Q = self.localVDN.forward(policy_q_vs, Aggd_Nei_policy_Q)
            Localized_Nei_target_Q = self.localVDN.forward(target_q_vs, Aggd_Nei_target_Q)
            loss = F.mse_loss(Localized_Nei_policy_Q, Localized_Nei_target_Q)

            return_lst.append(torch.mean(Localized_Nei_policy_Q))
            return_lst.append(loss.item())

        else:
            loss = F.mse_loss(policy_q_vs, target_q_vs)
            
            return_lst.append(torch.mean(policy_q_vs))
            return_lst.append(loss.item())

        self.opt.zero_grad()
        loss.backward()
        
        self.opt.step()
        if soft_copy:
            # update target network via soft copy with ratio tau
            # θ_target = τ*θ_local + (1 - τ)*θ_target
            for tp, lp in zip(self.target_model.parameters(), self.policy_model.parameters()):
                tp.data.copy_(self.tau*lp.data + (1.0-self.tau)*tp.data)
        else:
            # update target network via hard copy
            self.target_model.load_state_dict(self.policy_model.state_dict())
        # print("Back-propagation ends at ",datetime.now())

        return return_lst

    def getQValues(self, batch_idx):
        vec_states, img_states, actions, rewards, next_vec_states, next_img_states = self.memory.sampleByIndex(batch_idx)
        mod_img_states = np.full((self.batch_size, self.n_image_channel, self.x_len * 2, self.y_len * 2), 0)
        mod_next_img_states = np.full((self.batch_size, self.n_image_channel, self.x_len * 2, self.y_len * 2), 0)
        for i in range(self.batch_size):
            mod_img_states[i] = self.prepare_map(img_states[i], vec_states[i][0], vec_states[i][1])
            mod_next_img_states[i] = self.prepare_map(next_img_states[i], next_vec_states[i][0], next_vec_states[i][1])
        # target side

        # print("\nlearn: ")
        # print(vec_states)
        # img = im.fromarray(np.hstack(img_states).astype(np.float32) * 255)
        # img.show()
        # print(actions)
        # print(rewards)
        # print(next_vec_states)
        # img = im.fromarray(np.hstack(next_img_states).astype(np.float32) * 255)
        # img.show()

        vec_states = torch.tensor(vec_states).float().to(self.device)
        img_states = torch.tensor(mod_img_states).float().to(self.device)
        actions = torch.tensor(actions).long().to(self.device)
        rewards = torch.tensor(rewards).reshape(-1, 1).float().to(self.device)
        next_vec_states = torch.tensor(next_vec_states).float().to(self.device)
        next_img_states = torch.tensor(mod_next_img_states).float().to(self.device)

        _, next_idx = self.policy_model(next_vec_states, next_img_states).max(1)
        target_next_action_vs = self.target_model(next_vec_states, next_img_states).gather(1, next_idx.unsqueeze(1))
        target_q_vs = rewards + (self.gamma * target_next_action_vs)
        policy_q_vs = self.policy_model(vec_states, img_states).gather(1, actions.unsqueeze(1))

        return (policy_q_vs, target_q_vs)

    def save_models(self):
        self.policy_model.save_checkpoint()
        self.target_model.save_checkpoint()

    def load_models(self, load_path, id):
        self.model_path = load_path
        self.policy_model.load_checkpoint(load_path, "policy_model_" + str(id))
        self.target_model.load_checkpoint(load_path, "target_model_" + str(id))

    def copy_models(self):
        return ( self.policy_model.copy_policy(), self.target_model.copy_policy())

    def paste_models(self, new_policy):
        self.policy_model.paste_policy(new_policy[0])
        self.target_model.paste_policy(new_policy[1])

    def get_mem_cntr(self):
        return self.memory.size()
    