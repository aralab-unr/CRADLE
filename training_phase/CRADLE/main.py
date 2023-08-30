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
    n_episodes = 2500 # 2000
    n_timestamp = 320 # 320
    learning_rate = 0.0003
    batch_size = 8 # 8
    kick_start_timesteps = 12

    n_timestamp = n_timestamp + kick_start_timesteps

    n_drones = 3 # 3
    n_vector_obs = 2 + (4 * (n_drones-1))
    n_image_obs = (x_len, y_len)
    n_image_channel = 2
    n_actions = 4
    desired_comm_dist = 20

    gamma = 0.9
    tau = 0.001
    memory_size = 100000
    update_step = 1

    e1 = 80000 # 80000
    e2 = 800000 # 800000
    e3 = 3200000 # 3200000

    frame_num = 0
    miss_history = []
    reward_history = []
    coverage_history = []
    # loss_history = []
    least_miss = np.full(n_drones, 1.0)

    model_path = os.path.join('models', "t" + str(datetime.now()))

    uavs = [UAV(x_len, y_len, n_drones, n_vector_obs, n_image_obs, n_image_channel, n_actions, 
                batch_size, memory_size, update_step, learning_rate, gamma, tau, model_path, id, 
                desired_comm_dist) for id in range(n_drones)]
    
    load_path = "" # "t2023-07-05 18:35:52.268716"      * * * Enter the Folder Name to copy the models and other score histories * * * 

    if load_path == "":
        if not os.path.exists(model_path):
            os.makedirs(model_path)
    else:
        load_path = os.path.join('models', load_path)
        for id in range(n_drones):
            uavs[id].agent.load_models(load_path, id)
        print(' ... loaded model for all Drones ... ')
        model_path = load_path

        miss_history = np.loadtxt(os.path.join(model_path, "miss_history.csv"), delimiter=",", dtype=np.float32)
        least_miss = miss_history.min(axis=0)
        miss_history = miss_history.tolist()

        reward_history = np.loadtxt(os.path.join(model_path, "reward_history.csv"), delimiter=",", dtype=np.float32)
        reward_history = reward_history.tolist()

        coverage_history = np.loadtxt(os.path.join(model_path, "coverage_history.csv"), delimiter=",", dtype=np.float32)
        coverage_history = coverage_history.tolist()

        # loss_history = np.loadtxt(os.path.join(model_path, "loss_history.csv"), delimiter=",", dtype=np.float32)
        # loss_history = loss_history.tolist()


    actions = ["North", "East", "South", "West"]

    env = FireEnvironment()

    for episode in range(n_episodes):
        # To generate random positions for all UAVs
        for uav in uavs:
            uav.reset()

        env.reset(uavs)

        for _ in range(kick_start_timesteps):
            env.simStep()

        Mt = []
        fire_map_count = 0
        rew_o_ep = []

        fire_map_coverage = np.full((x_len, y_len), 0)

        for ti in range(n_timestamp - kick_start_timesteps):
            if ti % 4 == 0 :
                env.simStep()

            Mi_all_drones = []
            rew_o_ts = []
            coll_bmap = np.full((x_len, y_len), 0)

            fire_map = env.get_fire_map()
            fire_map_count += np.sum(fire_map)

            frame_num += 1

            for i in range(n_drones):

                if frame_num <= e1:
                    epsilon = 1
                elif frame_num <= e2:
                    epsilon -= ((1.0 - 0.1) / (e2 - e1))
                elif frame_num <= e3:
                    epsilon -= (0.1 / (e3 - e2))
                else:
                    epsilon = 0

                uavs[i].search_neighbors([other_uav.share_position() for other_uav in uavs if (uavs[i].id != other_uav.share_position()["id"])])
                
                for uav2 in uavs:
                    if uavs[i].id != uav2.id:
                        if uav2.id in uavs[i].neighbors:
                            uavs[i].UpdateObsFromNeighbor(uav2.share_obs())
                        else:
                            uavs[i].UpdateObsToDefault(uav2.id)
                
                action_id = int( uavs[i].act(uavs[i].get_Vector_obs(), uavs[i].get_Image_obs(), epsilon=epsilon) )
                uavs[i].obs.action = action_id
                prevUAV_copy = copy.deepcopy(uavs[i].obs)
                new_uav_obs, reward, done, _ = env.step(uavs[i], action_id)
                uavs[i].step(prevUAV_copy.get_Vector_obs(), prevUAV_copy.get_Image_obs(), action_id, reward, new_uav_obs.get_Vector_obs(), new_uav_obs.get_Image_obs(), done)
                # obs[i] = new_obs # No need obs[i] = new_obs, as the copy is by reference.

                if uavs[i].check_train():
                    batch_idx = np.random.choice(min(uavs[i].get_mem_cntr(), memory_size), batch_size, replace=False)
                    list_of_Nei_Q = []
                    for nei_id in uavs[i].neighbors:
                        list_of_Nei_Q.append(uavs[nei_id].share_my_q_values(batch_idx))

                    uavs[i].learn(batch_idx, list_of_Nei_Q)

                curr_score = (np.sum((fire_map - new_uav_obs.get_Image_obs()[0]).clip(0)))
                Mi_all_drones.append(curr_score)
                rew_o_ts.append(reward)

                # Assume this part is carried out by a consensus method to generate unified fire coverage map.
            fire_map_coverage += uavs[0].get_Image_obs()[0]
            
            rew_o_ep.append(rew_o_ts)
            
            Mt.append(Mi_all_drones)
            if (episode % 10 == 0) and (frame_num > e1):
                env.render(uavs)
            
        # No policy sharing.
        # policy_update_idx = np.argmax(np.array(rew_o_ep).sum(axis=0))
        # for i in range(n_drones):
        #     if i != policy_update_idx:
        #         uavs[i].paste_models(uavs[policy_update_idx].copy_models())

        Mt = (Mt / fire_map_count)
        total_Mt = np.sum(Mt, axis=0)

        # loss_history.append([uav.get_loss_value() for uav in uavs])

        print('Episode ', episode, " FireMiss= {:.2f}".format(np.min(total_Mt) * 100), ' % Rew.Ov.Ep.= ', np.array(rew_o_ep).sum(axis=0), ' Coverage= {:.2f}%'.format((np.sum(fire_map_coverage.clip(max = 1))/len(env.universal_fire_set)) * 100))
        miss_history.append(total_Mt)
        coverage_history.append(np.sum(fire_map_coverage.clip(max = 1)) / len(env.universal_fire_set))
        reward_history.append(np.array(rew_o_ep).sum(axis=0))
        
        save_flag = 0
        for i in range(n_drones):
            if least_miss[i] > total_Mt[i]:
                least_miss[i] = total_Mt[i]
                uavs[i].save_models()
                print(' ... saved checkpoint for Drone' + str(i) +' - ' + str(least_miss[i]) + '... ')
                save_flag = 1

        if save_flag == 1:
            np.savetxt(os.path.join(model_path, "miss_history.csv"), np.array(miss_history), delimiter=",", fmt='%.4e')
            np.savetxt(os.path.join(model_path, "reward_history.csv"), np.array(reward_history), delimiter=",", fmt='%.4e')
            np.savetxt(os.path.join(model_path, "coverage_history.csv"), np.array(coverage_history), delimiter=",", fmt='%.4e')
            # np.savetxt(os.path.join(model_path, "loss_history.csv"), np.array(loss_history), delimiter=",", fmt='%.4e')
            save_flag = 0

    for i in range(n_drones):
        uavs[i].save_models()
        print(' ... saved final checkpoint for Drone' + str(i) + ' ... ')
    np.savetxt(os.path.join(model_path, "miss_history.csv"), np.array(miss_history), delimiter=",", fmt='%.4e')
    np.savetxt(os.path.join(model_path, "reward_history.csv"), np.array(reward_history), delimiter=",", fmt='%.4e')
    np.savetxt(os.path.join(model_path, "coverage_history.csv"), np.array(coverage_history), delimiter=",", fmt='%.4e')
    # np.savetxt(os.path.join(model_path, "loss_history.csv"), np.array(loss_history), delimiter=",", fmt='%.4e')