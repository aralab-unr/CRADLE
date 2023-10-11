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

if __name__ == '__main__':

    start = time.time()
    x_len = 100
    y_len = 100
    n_episodes = 3000 # 2000
    n_timestamp = 320 # 320
    ep_start = 0
    learning_rate = 0.0003
    batch_size = 16 # 8
    kick_start_timesteps = 12

    n_timestamp = n_timestamp + kick_start_timesteps

    n_drones = 3 # 3
    n_vector_obs = 2 + (4 * (n_drones-1))
    n_image_obs = (x_len, y_len)
    n_image_channel = 2
    n_actions = 4
    desired_comm_dist = 30
    r_collision_avoidance = 20

    gamma = 0.9
    tau = 0.001
    memory_size = 100000
    update_step = 4
    model_saving_step = 100
    weighted_sum = 0.9

    # Initial epsilon
    # epsilon = 1.0
    # e1 = 80000 # 80000
    # e2 = 800000 # 800000
    # e3 = 3200000 # 3200000

    epsilon = 1.0
    e1 = 80000 # 80000
    e2 = 800000 # 800000
    e3 = 3200000 # 3200000

    frame_num = 0
    miss_history = []
    coll_miss_history = []
    reward_history = []
    coverage_history_indi = []
    coverage_history = []
    loss_history = []
    q_history = []
    q_tr_history = []
    q_tot_history = []

    model_path = os.path.join('models', ("t" + str(datetime.now())).replace(":", "_"))

    uavs = [UAV(x_len, y_len, n_drones, n_vector_obs, n_image_obs, n_image_channel, n_actions, 
                batch_size, memory_size, update_step, learning_rate, gamma, tau, weighted_sum, model_path, id, 
                desired_comm_dist) for id in range(n_drones)]
    
    load_path = "" # "t2023-07-05 18_35_52.268716"      * * * Enter the Folder Name to copy the models and other score histories * * * 

    if load_path == "":
        if not os.path.exists(model_path):
            os.makedirs(model_path)
            os.makedirs(os.path.join(model_path, "figureData"))
    else:
        load_path = os.path.join('models', load_path)
        for id in range(n_drones):
            uavs[id].agent.load_models(load_path, id)
        print(' ... loaded model for all Drones ... ')
        model_path = load_path

        miss_history = np.loadtxt(os.path.join(model_path, "miss_history.csv"), delimiter=",", dtype=np.float32)
        miss_history = miss_history.tolist()
        
        coll_miss_history = np.loadtxt(os.path.join(model_path, "coll_miss_history.csv"), delimiter=",", dtype=np.float32)
        coll_miss_history = coll_miss_history.tolist()

        reward_history = np.loadtxt(os.path.join(model_path, "reward_history.csv"), delimiter=",", dtype=np.float32)
        reward_history = reward_history.tolist()

        coverage_history_indi = np.loadtxt(os.path.join(model_path, "coverage_history_indi.csv"), delimiter=",", dtype=np.float32)
        coverage_history_indi = coverage_history_indi.tolist()

        coverage_history = np.loadtxt(os.path.join(model_path, "coverage_history.csv"), delimiter=",", dtype=np.float32)
        coverage_history = coverage_history.tolist()

        loss_history = np.loadtxt(os.path.join(model_path, "loss_history.csv"), delimiter=",", dtype=np.float32)
        loss_history = loss_history.tolist()

        q_history = np.loadtxt(os.path.join(model_path, "q_history.csv"), delimiter=",", dtype=np.float32)
        q_history = q_history.tolist()

        q_tr_history = np.loadtxt(os.path.join(model_path, "q_tr_history.csv"), delimiter=",", dtype=np.float32)
        q_tr_history = q_tr_history.tolist()

        q_tot_history = np.loadtxt(os.path.join(model_path, "q_tot_history.csv"), delimiter=",", dtype=np.float32)
        q_tot_history = q_tot_history.tolist()

        frame_num = len(miss_history) * 320
        if frame_num <= e1:
            epsilon = 1.0
        elif frame_num <= e2:
            epsilon = (1 - (((1.0 - 0.1) / (e2 - e1)) * (frame_num - e1)))
        elif frame_num <= e3:
            epsilon = (0.1 - ((0.1 / (e3 - e2)) * (frame_num - e2)))
        else:
            epsilon = 0.0
        ep_start = len(miss_history)
        print("Starting at " + str(ep_start) + " ep, frame_num=" + str(frame_num) + ", epsilon="+str(epsilon))

    env = FireEnvironment(x_len, y_len, r_collision_avoidance)

    for episode in range(ep_start, n_episodes):
        # To generate random positions for all UAVs
        for uav in uavs:
            uav.reset()

        env.reset(uavs)

        for _ in range(kick_start_timesteps):
            env.simStep()

        env.updateObsAfterReset(uavs)

        Mt = []
        fire_map_count = 0
        rew_o_ts = []
        q_o_ts = [[] for _ in range(n_drones)]
        q_tr_o_ts = []
        q_tot_o_ts = []
        loss_o_ts = []

        fire_map_coverage = np.full((x_len, y_len), 0, dtype=np.int32)
        fire_map_coverage_indi = [np.full((x_len, y_len), 0, dtype=np.int32) for _ in range(n_drones)]

        for ti in range(n_timestamp - kick_start_timesteps):
            if ti % 4 == 0 :
                env.simStep()

            Mi_all_drones = []
            rew_o_n = []
            q_tr_o_n = []
            q_tot_o_n = []
            loss_o_n = []

            fire_map = env.get_fire_map()
            fire_map_count += np.sum(fire_map)
            coll_coverage_ts = np.full((x_len, y_len), 0, dtype=np.int32)

            frame_num += 1

            # Search neighbors, communicate, collect their obs and update self relative obs.
            for i in range(n_drones):
                uavs[i].search_neighbors(uavs)

            # Consensus images with neighbors
            for i in range(n_drones):
                uavs[i].ConsensusWithNeiMaps()

            for i in range(n_drones):

                if frame_num <= e1:
                    epsilon = 1.0
                elif frame_num <= e2:
                    epsilon -= ((1.0 - 0.1) / (e2 - e1))
                elif frame_num <= e3:
                    epsilon -= (0.1 / (e3 - e2))
                else:
                    epsilon = 0.0
                
                if np.random.random() > epsilon:
                    Q_values = uavs[i].act(uavs[i].get_Vector_obs(), uavs[i].get_Image_obs())
                    q_o_ts[i].append(np.max(Q_values))
                    action_id = int( np.argmax(Q_values) )
                else:
                    action_id = np.random.randint(n_actions)
                
                uavs[i].obs.action = action_id
                prevUAV_copy = copy.deepcopy(uavs[i].obs)
                new_uav, reward, _, _ = env.step(uavs[i], action_id)
                # obs[i] = new_obs # No need obs[i] = new_obs, as the copy is by reference.
                # To update relative observation of new_uav w.r.to neighbor uav.
                for nei_obs in new_uav.neighbors.values():
                    new_uav.UpdateObsWithNeighborsInfo()

                uavs[i].step(prevUAV_copy.get_Vector_obs(), prevUAV_copy.get_Image_obs(), action_id, reward, new_uav.get_Vector_obs(), new_uav.get_Image_obs())

                if uavs[i].check_train(frame_num):
                    batch_idx = np.random.choice(min(uavs[i].get_mem_cntr(), memory_size), batch_size, replace=False)
                    list_of_Nei_Q = []
                    for nei_id in uavs[i].neighbors.keys():
                        list_of_Nei_Q.append(uavs[nei_id].share_my_q_values(batch_idx))

                    training_metrics = uavs[i].learn(batch_idx, list_of_Nei_Q)
                    q_tr_o_n.append(training_metrics[0].cpu().detach().numpy())
                    q_tot_o_n.append(training_metrics[1].cpu().detach().numpy())
                    loss_o_n.append(training_metrics[2])

                curr_score = (np.sum((fire_map - new_uav.get_Image_obs()[0]).clip(0)))
                Mi_all_drones.append(curr_score)
                rew_o_n.append(reward)

                # Assume this part is carried out by a consensus method to generate unified fire coverage map.
                fire_map_coverage += uavs[i].get_Image_obs()[0]
                fire_map_coverage_indi[i] += uavs[i].get_Image_obs()[0]
                coll_coverage_ts += uavs[i].get_Image_obs()[0]
            
            rew_o_ts.append(rew_o_n)
            if len(q_tr_o_n) > 0:
                q_tr_o_ts.append(q_tr_o_n)
            if len(q_tot_o_n) > 0:
                q_tot_o_ts.append(q_tot_o_n)
            if len(loss_o_n) > 0:
                loss_o_ts.append(loss_o_n)
            
            Mt.append(Mi_all_drones)
            if (episode % 10 == 0) and (frame_num > e1):
                env.render(uavs)
            
            coll_Mt_numr = (np.sum((fire_map - coll_coverage_ts.clip(max=1)).clip(0)))

        # # No policy sharing.
        # policy_update_idx = np.argmax(np.array(rew_o_ts).sum(axis=0))
        # for i in range(n_drones):
        #     if i != policy_update_idx:
        #         uavs[i].paste_models(uavs[policy_update_idx].copy_models())

        Mt = (Mt / fire_map_count)
        total_Mt = np.sum(Mt, axis=0)
        coll_Mt = (coll_Mt_numr / fire_map_count)
        coll_score_metric = np.sum(coll_Mt)

        print('Episode ', episode, " FireMiss= {:.2f}".format(np.min(total_Mt) * 100), ' % Rew.Ov.Ep.= ', np.array(rew_o_ts).sum(axis=0).astype(np.int32), ' Coverage= {:.2f}%'.format((np.sum(fire_map_coverage.clip(max = 1))/len(env.universal_fire_set)) * 100))
        
        # if ((episode+1) % 50 == 0):
        #     original_fire_map = np.full([x_len, y_len], 0, dtype=np.int8)
        #     for env_x, env_y in env.universal_fire_set:
        #             original_fire_map[env_x][env_y] = 1
        #     with open(os.path.join( model_path, "figureData", "data_ep"+str(episode)), "wb") as f:
        #         pickle.dump([original_fire_map, fire_map_coverage.clip(max = 1), [fire_map_coverage_indi[i].clip(max = 1) for i in range(n_drones)], str(np.sum(fire_map_coverage.clip(max = 1))) +",\n"+ str(len(env.universal_fire_set)) +",\n"+ "{:.2f}%".format((np.sum(fire_map_coverage.clip(max = 1))/len(env.universal_fire_set)))], f)

        miss_history.append(total_Mt)
        coll_miss_history.append(coll_score_metric)
        coverage_history_indi.append([np.sum(fire_map_coverage_indi[i].clip(max = 1)) / len(env.universal_fire_set) for i in range(n_drones)])
        coverage_history.append(np.sum(fire_map_coverage.clip(max = 1)) / len(env.universal_fire_set))
        reward_history.append(np.array(rew_o_ts).sum(axis=0))
        q_history.append([(sum(q_o_ts[i])/len(q_o_ts[i]) if len(q_o_ts[i])>0 else -1.0) for i in range(n_drones)])
        q_tr_history.append(np.mean(q_tr_o_ts, axis = 0))
        q_tot_history.append(np.mean(q_tot_o_ts, axis = 0))
        loss_history.append(np.mean(loss_o_ts, axis = 0))
        
        if (episode+1) % model_saving_step == 0 :
            for i in range(n_drones):
                uavs[i].save_models()
            np.savetxt(os.path.join(model_path, "miss_history.csv"), np.array(miss_history), delimiter=",", fmt='%.4e')
            np.savetxt(os.path.join(model_path, "coll_miss_history.csv"), np.array(coll_miss_history), delimiter=",", fmt='%.4e')
            np.savetxt(os.path.join(model_path, "reward_history.csv"), np.array(reward_history), delimiter=",", fmt='%.4e')
            np.savetxt(os.path.join(model_path, "coverage_history_indi.csv"), np.array(coverage_history_indi), delimiter=",", fmt='%.4e')
            np.savetxt(os.path.join(model_path, "coverage_history.csv"), np.array(coverage_history), delimiter=",", fmt='%.4e')
            np.savetxt(os.path.join(model_path, "q_history.csv"), np.array(q_history), delimiter=",", fmt='%.4e')
            np.savetxt(os.path.join(model_path, "q_tr_history.csv"), np.array(q_tr_history), delimiter=",", fmt='%.4e')
            np.savetxt(os.path.join(model_path, "q_tot_history.csv"), np.array(q_tot_history), delimiter=",", fmt='%.4e')
            np.savetxt(os.path.join(model_path, "loss_history.csv"), np.array(loss_history), delimiter=",", fmt='%.4e')
            
            print(' ... saved checkpoint ... ')

    end = time.time()
    total_time = end-start
    print("Total Time taken: ",total_time)
    with open(os.path.join( model_path, "figureData", "tr_duration"), "wb") as Tp:
        pickle.dump(total_time, Tp)

    # with open(os.path.join( model_path, "config_info.txt"), "a") as myfile:
    #     myfile.write("FileName: "+str(fileName)+" : ACM, Time taken: "+str(total_time)+"\n | gridWidth: "+str(gridWidth)+" | gridHeight: "+str(gridHeight)+
    #             " | playMode: "+str(playMode)+" | noTarget: "+str(noTarget)+" | noAgent: "+str(noAgent)+
    #             " | noObs: "+str(noObs)+" | noFreeway: "+str(noFreeway)+
    #             " | neighborWeights: "+str(neighborWeights)+" | totalEpisode: "+str(totalEpisode)+" | gamma: "+str(gamma)+
    #             " | epsilon: "+str(intEpsilon)+" | decay: "+str(decay)+" | alpha: "+str(alpha)+
    #             " | obsReward: "+str(obsReward)+" | freewayReward: "+str(freewayReward)+" | emptycellReward: "+str(emptycellReward)+
    #             " | hitwallReward: "+str(hitwallReward)+" | Attacker: "+str(Attacker)+" | Notes: "+str("with Attack and RAMPART")+"\n\n\n")