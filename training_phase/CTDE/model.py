import os
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class DDQN_Graph(nn.Module): 
    def __init__(self, n_vector_states, n_image_channel, n_actions, model_path, name): 
        super(DDQN_Graph, self).__init__()
        self.n_actions = n_actions
        self.checkpoint_file = os.path.join(model_path, name)

        # Vector hidden representation
        self.bn0 = nn.BatchNorm1d(1)
        self.dense_layer_1 = nn.Linear(n_vector_states, 50)
        self.dense_layer_2 = nn.Linear(50, 50)
        self.dense_layer_3 = nn.Linear(50, 50)

        # Image hidden representation
        self.bn1 = nn.BatchNorm2d(n_image_channel)
        # self.conv1 = nn.Conv2d(n_image_channel, 32, 3, stride=1)
        # self.conv2 = nn.Conv2d(32, 32, 3, stride=1)
        # self.conv3 = nn.Conv2d(32, 32, 3, stride=1)
        # self.conv4 = nn.Conv2d(32, 32, 3, stride=1)

        # self.conv1 = nn.Conv2d(n_image_channel, 4, 9, stride=1)
        # self.conv2 = nn.Conv2d(4, 8, 7, stride=1)
        # self.conv3 = nn.Conv2d(8, 16, 5, stride=1)
        # self.conv4 = nn.Conv2d(16, 32, 3, stride=1)

        self.conv1 = nn.Conv2d(n_image_channel, 4, 3, stride=1)
        self.conv2 = nn.Conv2d(4, 8, 3, stride=1)
        self.conv3 = nn.Conv2d(8, 12, 3, stride=1)
        self.conv4 = nn.Conv2d(12, 16, 3, stride=1)

        self.maxpool1 = nn.MaxPool2d(2, stride=2)

        # Concatinated hidden representation
        self.main_dense_layer = nn.Linear(1650, 256)
        self.a_layer = nn.Linear(256, n_actions)
        
    
    def forward(self, vector_in, image_in):
        vector_in = vector_in.unsqueeze(1)
        vector_in = self.bn0(vector_in)
        vector_in = vector_in.squeeze(1)
        vec = F.relu(self.dense_layer_1(vector_in))
        vec = F.relu(self.dense_layer_2(vec))
        vec = F.relu(self.dense_layer_3(vec))

        img = self.bn1(image_in)
        img = self.conv1(img)
        img = self.maxpool1(img)
        img = self.conv2(img)
        img = self.maxpool1(img)
        img = self.conv3(img)
        img = self.maxpool1(img)
        img = self.conv4(img)
        img = self.maxpool1(img)

        img = img.view(img.size()[0], -1)

        full_out = torch.cat((vec, img), dim = 1)

        c = F.relu(self.main_dense_layer(full_out))
        actions_q = self.a_layer(c)

        return actions_q
    
    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self, load_path, load_name):
        self.checkpoint_file = os.path.join(load_path, load_name)
        self.load_state_dict(torch.load(self.checkpoint_file))

    def copy_policy(self):
        return self.state_dict()
    
    def paste_policy(self, new_policy):
        self.load_state_dict(new_policy)

class ReplayMemory(): 
    def __init__(self, max_size, vec_shape, img_shape, n_image_channel, n_drone): 
        super(ReplayMemory, self).__init__() 
        self.mem_size = max_size
        self.n_drone = n_drone
        self.mem_cntr = 0
        self.vec_state_memory = np.zeros((self.mem_size, self.n_drone, vec_shape),
                                     dtype=np.float32)
        self.img_state_memory = np.zeros((self.mem_size, self.n_drone, n_image_channel, img_shape[0], img_shape[1]),
                                     dtype=np.float32)
        self.action_memory = np.zeros((self.mem_size, self.n_drone), dtype=np.float32)
        self.reward_memory = np.zeros((self.mem_size, self.n_drone), dtype=np.float32)
        self.new_vec_state_memory = np.zeros((self.mem_size, self.n_drone, vec_shape),
                                         dtype=np.float32)
        self.new_img_state_memory = np.zeros((self.mem_size, self.n_drone, n_image_channel, img_shape[0], img_shape[1]),
                                        dtype=np.float32)
        self.terminal_memory = np.zeros((self.mem_size, self.n_drone), dtype=np.float32)
    
    def size(self):
        return self.mem_cntr
    
    def push(self, vec_state, img_state, action, reward, vec_state_, img_state_, done):
        index = self.mem_cntr % self.mem_size
        self.vec_state_memory[index] = vec_state
        self.img_state_memory[index] = img_state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.new_vec_state_memory[index] = vec_state_
        self.new_img_state_memory[index] = img_state_
        self.terminal_memory[index] = done
        self.mem_cntr += 1
    
    def sample(self, batch_size):

        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        vec_states = self.vec_state_memory[batch]
        img_states = self.img_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        next_vec_states = self.new_vec_state_memory[batch]
        next_img_states = self.new_img_state_memory[batch]
        terminal = self.terminal_memory[batch]
        
        return (vec_states, img_states, actions, rewards, next_vec_states, next_img_states, terminal)
    
    def __len__(self):
        return self.max_size