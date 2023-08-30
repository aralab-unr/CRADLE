import os
import numpy as np
from environment import FireEnvironment
from UAVmodel import UAV

from PIL import Image as im
import copy

import sys
np.set_printoptions(threshold=sys.maxsize)
from datetime import datetime
import pickle


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
    wildfire_timeseries = []
    uav_coords_ts = []

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

        fire_map_coverage = np.full((x_len, y_len), 0)

        for ti in range(n_timestamp - kick_start_timesteps):
            if ti % 4 == 0 :
                env.simStep()

            wildfire_timeseries.append((env.fire_set.copy(), env.fire_off.copy()))

            uav_coords = []
            for i in range(n_drones):
                
                uav_coords.append((uavs[i].obs.vector.pos.x, uavs[i].obs.vector.pos.y))
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

                # Assume this part is carried out by a consensus method to generate unified fire coverage map.
                fire_map_coverage += uavs[i].get_Image_obs()[0]
            
            env.render(uavs)
            uav_coords_ts.append(uav_coords)

        # Download fire_map_coverage.clip(max = 1)
        with open("figureData/uav_coords_ts", "wb") as f:
            pickle.dump(uav_coords_ts, f)

        # Download fire_map_coverage.clip(max = 1)
        with open("figureData/VDN_belief_map_matrix", "wb") as f:
            pickle.dump(fire_map_coverage.clip(max = 1), f)

        # Download real_fire_map
        real_fire_map = np.full([x_len, y_len], 0)
        for env_x, env_y in env.universal_fire_set:
                real_fire_map[env_x][env_y] = 1
        with open("figureData/VDN_real_fire_map_matrix", "wb") as f:
            pickle.dump(real_fire_map, f)

    # Download wildfire_timeseries
    with open("figureData/VDN_wildfire_timeseries", "wb") as f:
        pickle.dump(wildfire_timeseries, f)