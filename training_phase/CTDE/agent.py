import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from model import DDQN_Graph, ReplayMemory

from datetime import datetime
from PIL import Image as im

class VDN(nn.Module):
    def __init__(self):
        super(VDN, self).__init__()

    def forward(self, q_values):
        # Sum up the Q-values of all agents
        return q_values.sum(dim=0)

class DDQN_Agent(): 
    def __init__(self, x_len, y_len, n_drones, n_vector_obs, n_image_obs, n_image_channel, n_actions, batch_size, memory_size, 
                 update_step, learning_rate, gamma, tau, model_path):
        super(DDQN_Agent, self).__init__()
        self.x_len = x_len
        self.y_len = y_len
        self.n_drones = n_drones
        self.n_vector_obs = n_vector_obs
        self.n_image_obs = n_image_obs
        self.n_image_channel = n_image_channel
        self.n_actions = n_actions
        self.batch_size = batch_size
        self.update_step = update_step
        self.lr = learning_rate
        self.gamma = gamma
        self.tau = tau
        self.model_path = model_path
        self.setup_gpu()
        self.setup_model()
        self.setup_opt()
        self.memory = ReplayMemory(memory_size, self.n_vector_obs, self.n_image_obs, self.n_image_channel, self.n_drones)
        self.prepare_train()

        self.vdn = VDN()

        self.loss_value = 0.0
    
    def setup_gpu(self): 
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def setup_model(self):
        self.policy_model = [DDQN_Graph(
            self.n_vector_obs,
            self.n_image_channel, 
            self.n_actions,
            self.model_path,
            'policy_model_' + str(id)).to(self.device) for id in range(self.n_drones)]
        self.target_model = [DDQN_Graph(
            self.n_vector_obs,
            self.n_image_channel, 
            self.n_actions,
            self.model_path,
            'target_model_' + str(id)).to(self.device) for id in range(self.n_drones)]
    
    def setup_opt(self):
        self.opt = torch.optim.Adam(self.get_parameters(), lr=self.lr)

    def get_parameters(self):
        # Collect parameters from all agents
        for model in self.policy_model:
            yield from model.parameters()
    
    def prepare_train(self):
        self.steps = 0

    def prepare_map(self, vec_in, maps_in):
        ext_x_lim = self.x_len * 2
        ext_y_lim = self.y_len * 2
        ext_x_mid = ext_x_lim//2
        ext_y_mid = ext_y_lim//2

        map_out = []
        for idx, map_in in enumerate(maps_in):
            x, y = vec_in[idx]
            main_bm = np.full((self.n_image_channel, ext_x_lim, ext_x_lim), 0).astype(np.float32)
            for idx2 in range(self.n_image_channel):
                main_bm[idx2, (ext_x_mid-int(x)):(ext_x_mid + (ext_x_mid-int(x))), (ext_y_mid-int(y)):(ext_y_mid + (ext_y_mid-int(y)))] = map_in[idx2]
            map_out.append(main_bm)
        # im.fromarray(np.hstack(map_out).astype(np.float32) * 255).show()
        return np.array(map_out)
    
    def act(self, vector_obs, image_obs, epsilon):
        mod_image_obs = self.prepare_map(vector_obs[:,0:2], image_obs)

        actions_list = []
        for idx, mod_img in enumerate(mod_image_obs):
            vec_obs_tensor = torch.tensor(np.array([vector_obs[idx]])).float().to(self.device)
            mod_img_tensor = torch.tensor(np.array([mod_img])).float().to(self.device)
            for idx in range(self.n_drones):
                self.policy_model[idx].eval()
                with torch.no_grad(): 
                    action_vs = self.policy_model[idx](vec_obs_tensor, mod_img_tensor)
                self.policy_model[idx].train()
            if np.random.random() > epsilon:
                actions_list.append(np.argmax(action_vs.cpu().detach().numpy()))
            else:
                actions_list.append(np.random.randint(self.n_actions))
        
        return np.array(actions_list)
    
    def step(self, cur_vec_state, cur_img_state, action, reward, next_vec_state, next_img_state, done):
        self.memory.push(cur_vec_state, cur_img_state, action, reward, next_vec_state, next_img_state, done)
        self.steps = (self.steps + 1) % self.update_step
        if self.steps == 0 and self.memory.size() >= self.batch_size:
            # print("Back-propagation starts at ",datetime.now())
            exps = self.memory.sample(self.batch_size)
            self.learn(exps)
            # print("Back-propagation ends at ",datetime.now())
        else:
            pass
    
    def learn(self, exps, soft_copy=True):
        
        # states: batch_size, state_size
        # actions: batch_size, 1
        # rewards: batch_size, 1
        # next_states: batch_size, state_size
        # dones: batch_size, 1
        vec_states, img_states, actions, rewards, next_vec_states, next_img_states, dones = exps
        
        mod_img_states = np.full((self.batch_size, self.n_drones, self.n_image_channel, self.x_len*2, self.y_len*2), 0)
        mod_next_img_states = np.full((self.batch_size, self.n_drones, self.n_image_channel, self.x_len*2, self.y_len*2), 0)
        for i in range(self.batch_size):
            mod_img_states[i] = self.prepare_map(vec_states[i][:,0:2], img_states[i])
            mod_next_img_states[i] = self.prepare_map(next_vec_states[i][:,0:2], next_img_states[i])

        rewards = torch.tensor(rewards.T).float().to(self.device)
        
        curr_q_list = []
        target_q_list = []
        for i in range(self.n_drones):
            vec_state = vec_states[:,i,:]
            mod_img_state = mod_img_states[:,i,:,:,:]
            next_vec_state = next_vec_states[:,i,:]
            mod_next_img_state = mod_next_img_states[:,i,:,:,:]

            vec_state = torch.tensor(vec_state).float().to(self.device)
            mod_img_state = torch.tensor(mod_img_state).float().to(self.device)
            action = torch.tensor(actions[:,i]).long().to(self.device)
            next_vec_state = torch.tensor(next_vec_state).float().to(self.device)
            mod_next_img_state = torch.tensor(mod_next_img_state).float().to(self.device)

            policy_q_vs = self.policy_model[i](vec_state, mod_img_state).gather(1, action.unsqueeze(1))
            curr_q_list.append(policy_q_vs)

            _, next_idx = self.policy_model[i](next_vec_state, mod_next_img_state).max(1)
            target_next_action_vs = self.target_model[i](next_vec_state, mod_next_img_state).gather(1, next_idx.unsqueeze(1))
            target_q_vs = rewards[i].unsqueeze(1) + (self.gamma * target_next_action_vs)
            target_q_list.append(target_q_vs)

        loss = F.mse_loss(self.vdn(torch.stack(curr_q_list)), self.vdn(torch.stack(target_q_list)))

        self.loss_value = loss.item()

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        if soft_copy:
            # update target network via soft copy with ratio tau
            # θ_target = τ*θ_local + (1 - τ)*θ_target
            for idx in range(self.n_drones):
                for tp, lp in zip(self.target_model[idx].parameters(), self.policy_model[idx].parameters()):
                    tp.data.copy_(self.tau*lp.data + (1.0-self.tau)*tp.data)
        else:
            # update target network via hard copy
            for idx in range(self.n_drones):
                self.target_model[idx].load_state_dict(self.policy_model[idx].state_dict())

    def save_models(self):
        for idx in range(self.n_drones):
            self.policy_model[idx].save_checkpoint()
            self.target_model[idx].save_checkpoint()

    def load_models(self, load_path):
        self.model_path = load_path
        for i in range(self.n_drones):
            self.policy_model[i].load_checkpoint(load_path, "policy_model_" + str(i))
            self.target_model[i].load_checkpoint(load_path, "target_model_" + str(i))

    def update_policy(self, policy_update_idx):
        for i in range(self.n_drones):
            if i != policy_update_idx:
                self.paste_models(self.copy_models(policy_update_idx), i)

    def copy_models(self, idx):
        return ( self.policy_model[idx].copy_policy(), self.target_model[idx].copy_policy())

    def paste_models(self, new_policy, idx):
        self.policy_model[idx].paste_policy(new_policy[0])
        self.target_model[idx].paste_policy(new_policy[1])