import os
import numpy as np
from environment import FireEnvironment
from UAVmodel import UAV

# from PIL import Image as im
# import matplotlib.pyplot as plt
import copy

import sys
np.set_printoptions(threshold=sys.maxsize)
from datetime import datetime
import time
import pickle
import moviepy.editor as mpy
from scipy.ndimage import zoom

if __name__ == '__main__':

    x_len = 100
    y_len = 100
    n_samples = 1 # 10
    n_timestamp = 320 # 320
    kick_start_timesteps = 12

    n_timestamp = n_timestamp + kick_start_timesteps

    n_prev_drones = 3
    n_prev_vector_obs = 2 + (4 * (n_prev_drones-1))

    # n_drones_list = [3, 5, 7, 9, 11]
    n_drones_list = [3]
    n_image_obs = (x_len, y_len)
    n_image_channel = 2
    n_actions = 4
    desired_comm_dist = 30
    cov_thres_t = 0.80

    model_path = os.path.join('models', ("t" + str(datetime.now())).replace(":", "_"))
    
    load_path = "t2023-09-12 15_26_44.451624" # "t2023-07-05 18_35_52.268716"      * * * Enter the Folder Name to copy the models and other score histories * * * 
    model_path = os.path.join('models', load_path)

    env = FireEnvironment(x_len, y_len)

    coverage_o_n = []
    coverage_time_o_n = []

    for n_drones in n_drones_list:
    
        uavs = [UAV(x_len, y_len, n_drones, n_prev_vector_obs, n_image_obs, n_image_channel, n_actions, 
                model_path, id, desired_comm_dist) for id in range(n_drones)]

        trained_policy_id = np.random.randint(0, n_prev_drones, n_drones)
        for id in range(n_drones):
            uavs[id].agent.load_models(model_path, trained_policy_id[id])
        print(' ... loaded model for all Drones ... ')
        data_path = os.path.join('data', 'uav_'+str(n_drones))
        if not os.path.exists(data_path):
            os.makedirs(data_path)

        coverage_o_samples = []
        coverage_time_o_s = []

        for sample_i in range(n_samples):
            start = time.time()
            reached_80_flag = 0
            # To generate random positions for all UAVs
            for uav in uavs:
                uav.reset()

            env.reset(uavs)

            for _ in range(kick_start_timesteps):
                env.simStep()

            env.updateObsAfterReset(uavs)

            fire_map_coverage = np.full((x_len, y_len), 0, dtype=np.int32)

            for ti in range(n_timestamp - kick_start_timesteps):
                if ti % 4 == 0 :
                    env.simStep()

                # Search neighbors, communicate, collect their obs and update self relative obs.
                for i in range(n_drones):
                    uavs[i].search_neighbors(uavs)

                # Consensus images with neighbors
                for i in range(n_drones):
                    uavs[i].ConsensusWithNeiMaps()

                for i in range(n_drones):
                    
                    vec_obs = uavs[i].get_Vector_obs()
                    vec_obs_pos_x = vec_obs[0]
                    vec_obs_pos_y = vec_obs[1]
                    vec_obs_rel = vec_obs[2:]
                    Q_values_list = []
                    for rel_vec_i in range(0, len(vec_obs_rel), (n_prev_vector_obs-2)):
                        Q_values_list.append(uavs[i].act(np.hstack([vec_obs_pos_x, vec_obs_pos_y, vec_obs_rel[rel_vec_i:rel_vec_i+(n_prev_vector_obs-2)]]).astype(np.float32), uavs[i].get_Image_obs()))
                    action_id = int( np.argmax(np.array(Q_values_list).sum(axis = 0)))
                    uavs[i].obs.action = action_id
                    new_uav, reward, _, _ = env.step(uavs[i], action_id)
                    # obs[i] = new_obs # No need obs[i] = new_obs, as the copy is by reference.
                    # To update relative observation of new_uav w.r.to neighbor uav.
                    for nei_obs in new_uav.neighbors.values():
                        new_uav.UpdateObsWithNeighborsInfo()

                    # Assume this part is carried out by a consensus method to generate unified fire coverage map.
                    fire_map_coverage += uavs[i].get_Image_obs()[0]

                env.render(uavs)
                if ((np.sum(fire_map_coverage.clip(max = 1)) / len(env.universal_fire_set)) > cov_thres_t) and (reached_80_flag==0):
                    end = time.time()
                    reached_80_flag = 1

            img = []
            image = env.env_frames
            # for i in range(0, len(image), 4):
            #     img.append(image[i])
            img = image
            clip = mpy.ImageSequenceClip(img, durations=[1/20.0]*len(img))
            clip.fps = 20
            factor_x = 400 / clip.size[0]
            factor_y = 400 / clip.size[1]
            resized_images = [zoom(i, (factor_y, factor_x, 1), order=0) for i in img]
            # clip = clip.resize(newsize=(400, 400))
            clip = mpy.ImageSequenceClip(resized_images, durations=[1/20.0]*len(resized_images))
            clip.fps = 20
            clip.write_gif('env.gif')

            if reached_80_flag == 1:
                coverage_time_o_s.append(end-start)
            coverage_o_samples.append(np.sum(fire_map_coverage.clip(max = 1)) / len(env.universal_fire_set))
            print("n=",n_drones,", sample#=",sample_i,", Total coverage =", "{:.2f}%".format((np.sum(fire_map_coverage.clip(max = 1)) / len(env.universal_fire_set))*100))
        print("n=",n_drones,", ",coverage_o_samples, coverage_time_o_s, np.mean(coverage_o_samples), np.mean(coverage_time_o_s))
    #     np.savetxt(os.path.join(data_path, "coverage_o_s.csv"), coverage_o_samples, delimiter=",", fmt='%.4e')        
    #     np.savetxt(os.path.join(data_path, "coverage_time_o_s.csv"), coverage_time_o_s, delimiter=",", fmt='%.4e')
    #     coverage_o_n.append(np.mean(coverage_o_samples))
    #     coverage_time_o_n.append(np.mean(coverage_time_o_s))

    # np.savetxt(os.path.join("data", "avgd_coverage_o_n.csv"), coverage_o_n, delimiter=",", fmt='%.4e') 
    # np.savetxt(os.path.join("data", "coverage_time_o_n.csv"), coverage_time_o_n, delimiter=",", fmt='%.4e')        
    print(' ... saved checkpoint ... ')