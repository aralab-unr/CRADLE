import numpy as np
from NNagent import DDQN_Agent, Observation_space, Relative_position

class UAV:
    def __init__ (self, x_len, y_len, n_drones, n_vector_obs, n_image_obs, n_image_channel, n_actions, 
                  batch_size, memory_size, update_step, learning_rate, gamma, tau, weighted_sum, model_path, id, 
                  desired_comm_dist):
        super(UAV, self).__init__()
        self.id = id
        self.obs = Observation_space(x_len, y_len, n_drones)
        self.agent = DDQN_Agent(x_len, y_len, n_vector_obs, n_image_obs = n_image_obs, n_image_channel = n_image_channel, 
                                n_actions = n_actions, batch_size = batch_size, memory_size = memory_size, 
                                update_step = update_step, learning_rate = learning_rate, gamma = gamma, 
                                tau = tau, weighted_sum = weighted_sum, model_path = model_path, id = id)
        self.neighbors = {}
        self.desired_comm_dist = desired_comm_dist

    def get_Vector_obs(self):
        return self.obs.get_Vector_obs()
    
    def get_Image_obs(self):
        return self.obs.get_Image_obs()

    def reset(self):
        self.obs.reset()
        self.neighbors = {}

    def search_neighbors(self, other_uavs):
        nei = {}
        curr_pos = self.get_position()
        for other_uav in other_uavs:
            if self.id != other_uav.id:
                other_uav_pos = other_uav.get_position()
                if np.sqrt( (other_uav_pos[0] - curr_pos[0])**2 + (other_uav_pos[1] - curr_pos[1])**2 ) <= self.desired_comm_dist:
                    nei[other_uav.id] = other_uav.share_obs()
                    self.UpdateObsFromNeighbors(other_uav.share_obs())
                else:
                    self.UpdateObsToDefault(other_uav.id)
        self.neighbors = nei

    def get_position(self):
        return self.obs.share_position()
    
    def share_obs(self):
        return { "id" : self.id, "agent_observation" : self.obs.share_obs()}
    
    def UpdateObsFromNeighbors(self, Nei_obs):
        nei_id = Nei_obs["id"]
        nei_obs = Nei_obs["agent_observation"]

        if nei_id > self.id:
            nei_id = nei_id - 1

        self.obs.vector.rel_pos_j[nei_id] = Relative_position((nei_obs[0] - self.obs.vector.pos.x),(nei_obs[1] - self.obs.vector.pos.y))
        self.obs.vector.dist_j[nei_id] = np.sqrt( (nei_obs[0] - self.obs.vector.pos.x)**2 + (nei_obs[1] - self.obs.vector.pos.y)**2 )
        self.obs.vector.act_j[nei_id] = nei_obs[4]

    def UpdateObsWithNeighborsInfo(self):
        for Nei_obs in list(self.neighbors.values()):
            nei_id = Nei_obs["id"]
            nei_obs = Nei_obs["agent_observation"]

            if nei_id > self.id:
                nei_id = nei_id - 1

            self.obs.vector.rel_pos_j[nei_id] = Relative_position((nei_obs[0] - self.obs.vector.pos.x),(nei_obs[1] - self.obs.vector.pos.y))
            self.obs.vector.dist_j[nei_id] = np.sqrt( (nei_obs[0] - self.obs.vector.pos.x)**2 + (nei_obs[1] - self.obs.vector.pos.y)**2 )

    def ConsensusWithNeiMaps(self):
        for Nei_obs in list(self.neighbors.values()):
            nei_obs = Nei_obs["agent_observation"]
            cm_lim = nei_obs[2].shape
            for i in range(cm_lim[0]):
                for j in range(cm_lim[1]):
                    if self.obs.image.coverage_map[i][j] < nei_obs[3][i][j]:
                        self.obs.image.coverage_map[i][j] = nei_obs[3][i][j]
                        self.obs.image.belief_map[i][j] = nei_obs[2][i][j]  
    
    def UpdateObsToDefault(self, nei_id):

        if nei_id > self.id:
            nei_id = nei_id - 1

        self.obs.vector.rel_pos_j[nei_id] = Relative_position(-200, -200)
        self.obs.vector.dist_j[nei_id] = -1
        self.obs.vector.act_j[nei_id] = -1
    
    def act(self, vector_obs, mod_map):
        return self.agent.act(vector_obs, mod_map)
    
    def step(self, cur_vec_state, cur_img_state, action, reward, next_vec_state, next_img_state):
        self.agent.step(cur_vec_state, cur_img_state, action, reward, next_vec_state, next_img_state)

    def check_train(self, frame_num):
        return self.agent.check_train(frame_num)

    def share_my_q_values(self, batch_idx):
        return self.agent.getQValues(batch_idx)

    def learn(self, batch_idx, list_of_Nei_Q):

        Nei_policy_Q_list = []
        Nei_target_Q_list = []
        for Nei_policy_Q, Nei_target_Q in list_of_Nei_Q:
            Nei_policy_Q_list.append(Nei_policy_Q)
            Nei_target_Q_list.append(Nei_target_Q)

        return self.agent.learn(batch_idx, (Nei_policy_Q_list, Nei_target_Q_list))

    def save_models(self):
        self.agent.save_models()

    def load_models(self, load_path, id):
        self.agent.load_models(load_path, id)

    def copy_models(self):
        return self.agent.copy_models()

    def paste_models(self, new_policy):
        self.agent.paste_models(new_policy)

    def get_mem_cntr(self):
        return self.agent.get_mem_cntr()
    
    def get_loss_value(self):
        return self.agent.get_loss_value()