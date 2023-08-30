import os
import numpy as np
from environment import FireEnvironment
from UAVmodel import UAV

from PIL import Image as im
import copy

import sys
np.set_printoptions(threshold=sys.maxsize)
from datetime import datetime


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
    desired_comm_dist = 20

    Mt = []
    rewards = []
    coverage = []

    model_path = os.path.join('models', "t" + str(datetime.now()))

    uavs = [UAV(x_len, y_len, n_drones, n_vector_obs, n_image_obs, n_image_channel, n_actions, 
                model_path, id, desired_comm_dist) for id in range(n_drones)]
    
    load_path = "t2023-08-22 12:49:13.724368" # "t2023-07-05 18:35:52.268716"      * * * Enter the Folder Name to copy the models and other score histories * * * 

    load_path = os.path.join('models', load_path)
    for id in range(n_drones):
        uavs[id].agent.load_models(load_path, id)
    print(' ... loaded model for all Drones ... ')
    model_path = load_path
    print(model_path)

    actions = ["North", "East", "South", "West"]

    env = FireEnvironment()

    for episode in range(n_episodes):
        # To generate random positions for all UAVs
        for uav in uavs:
            uav.reset()
        env.reset(uavs)

        for _ in range(kick_start_timesteps):
            env.simStep()

        fire_map_count = 0

        fire_map_coverage = np.full((x_len, y_len), 0)

        for ti in range(n_timestamp - kick_start_timesteps):
            if ti % 4 == 0 :
                env.simStep()

            Mi_all_drones = []
            rew_o_ts = []

            fire_map = env.get_fire_map()
            fire_map_count += np.sum(fire_map)

            for i in range(n_drones):

                uavs[i].search_neighbors([other_uav.share_position() for other_uav in uavs if (uavs[i].id != other_uav.share_position()["id"])])
                
                for uav2 in uavs:
                    if uavs[i].id != uav2.id:
                        if uav2.id in uavs[i].neighbors:
                            uavs[i].UpdateObsFromNeighbor(uav2.share_obs())
                        else:
                            uavs[i].UpdateObsToDefault(uav2.id)
                
                action_id = int( uavs[i].act(uavs[i].get_Vector_obs(), uavs[i].get_Image_obs()) )
                uavs[i].obs.action = action_id
                prevUAV_copy = copy.deepcopy(uavs[i].obs)
                new_uav_obs, reward, done, _ = env.step(uavs[i], action_id)

                curr_score = (np.sum((fire_map - new_uav_obs.get_Image_obs()[0]).clip(0)))
                Mi_all_drones.append(curr_score)
                rew_o_ts.append(reward)

                # Assume this part is carried out by a consensus method to generate unified fire coverage map.
                fire_map_coverage += uavs[i].get_Image_obs()[0]
            
            rewards.append(rew_o_ts)
            Mt.append(Mi_all_drones)
            coverage.append(np.sum(fire_map_coverage.clip(max = 1)) / len(env.universal_fire_set))
            env.render(uavs)

        print("Done...")
        
        np.savetxt(os.path.join(model_path, "miss_history.csv"), np.array(Mt)/fire_map_count, delimiter=",", fmt='%.4e')
        np.savetxt(os.path.join(model_path, "reward_history.csv"), np.array(rewards), delimiter=",", fmt='%.4e')
        np.savetxt(os.path.join(model_path, "coverage_history.csv"), np.array(coverage), delimiter=",", fmt='%.4e')