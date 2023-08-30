import os
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import pickle

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

        self.conv1 = nn.Conv2d(n_image_channel, 2, 3, stride=1)
        self.conv2 = nn.Conv2d(2, 4, 3, stride=1)
        self.conv3 = nn.Conv2d(4, 8, 3, stride=1)
        self.conv4 = nn.Conv2d(8, 16, 3, stride=1)

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

        with open("figureData/VDN_image_in", "wb") as f:
            pickle.dump(image_in, f)
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
    
    def load_checkpoint(self, load_path, load_name):
        self.checkpoint_file = os.path.join(load_path, load_name)
        self.load_state_dict(torch.load(self.checkpoint_file))
