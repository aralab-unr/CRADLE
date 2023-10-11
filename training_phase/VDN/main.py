import os
import numpy as np
from environment import FireEnvironment
from agent import DDQN_Agent, Observation_space

from PIL import Image as im
import copy

import sys
np.set_printoptions(threshold=sys.maxsize)
from datetime import datetime
import time
import pickle

def ConsensusImageObs(obss):
    dom_bmap = obss[0].image.belief_map.copy()
    dom_cmap = obss[0].image.coverage_map.copy()
    cm_lim = dom_cmap.shape

    for obs in obss[1:]:
        for i in range(cm_lim[0]):
            for j in range(cm_lim[1]):
                if dom_cmap[i][j] < obs.image.coverage_map[i][j]:
                    dom_cmap[i][j] = obs.image.coverage_map[i][j]
                    dom_bmap[i][j] = obs.image.belief_map[i][j]

    for obs in obss:
        obs.image.coverage_map = dom_cmap.copy()
        obs.image.belief_map = dom_bmap.copy()

if __name__ == '__main__':

    start = time.time()
    x_len = 100
    y_len = 100
    n_episodes = 3000 # 3000
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
    r_collision_avoidance = 20

    gamma = 0.9
    tau = 0.001
    memory_size = 100000
    update_env_step = 4
    update_step = 4
    model_saving_step = 100

    # Initial epsilon
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

    model_path = os.path.join('models', "t" + str(datetime.now()).replace(":", "_"))

    agent = DDQN_Agent(x_len, y_len, n_drones, n_vector_obs, n_image_obs = n_image_obs, n_image_channel = n_image_channel, n_actions = n_actions, batch_size = batch_size, memory_size = memory_size, 
                 update_step = update_step, learning_rate = learning_rate, gamma = gamma, tau = tau, model_path = model_path)
    
    load_path = "" # "t2023-07-05 18:35:52.268716"      * * * Enter the Folder Name to copy the models and other score histories * * * 

    if load_path == "":
        if not os.path.exists(model_path):
            os.makedirs(model_path)
            os.makedirs(os.path.join(model_path, "figureData"))
    else:
        load_path = os.path.join('models', load_path)
        agent.load_models(load_path)
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
        env.reset()
        # Instance of agent which stores the agent's current observation info.
        obss = [Observation_space(i, x_len, y_len) for i in range(n_drones)]

        # This is important for the drones to navigate towards the fire.
        for i in range(n_drones):
            obss[i].image.belief_map = env.binary_val.copy()

        for _ in range(kick_start_timesteps):
            env.simStep()
        
        for i in range(n_drones):
            # obss[i].UpdateObsFromOthers(obss)
            _ = env.reward(obss[i], obss) # Initializing the obs's belief map & coverage map w.r.to currrent state fire map.

        Mt = []
        coll_Mt = []
        fire_map_count = 0
        rew_o_ts = []
        q_o_ts = [[] for _ in range(n_drones)]
        q_tr_o_ts = []
        q_tot_o_ts = []
        loss_o_ts = []

        fire_map_coverage = np.full((x_len, y_len), 0, dtype=np.int32)
        fire_map_coverage_indi = [np.full((x_len, y_len), 0, dtype=np.int32) for _ in range(n_drones)]

        for ti in range(n_timestamp - kick_start_timesteps):
            if ti % update_env_step == 0 :
                env.simStep()

            fire_map = env.get_fire_map()
            fire_map_count += np.sum(fire_map)
            coll_coverage_ts = np.full((x_len, y_len), 0, dtype=np.int32)

            frame_num += 1

            for i in range(n_drones):
                obss[i].UpdateObsFromOthers(obss)
            unchanged_obs = copy.deepcopy(obss)
            ConsensusImageObs(obss)

            if frame_num <= e1:
                epsilon = 1.0
            elif frame_num <= e2:
                epsilon -= ((1.0 - 0.1) / (e2 - e1))
            elif frame_num <= e3:
                epsilon -= (0.1 / (e3 - e2))
            else:
                epsilon = 0.0
            
            vector_observations = []
            image_observations = []
            for obs in obss:
                vector_observations.append(obs.get_Vector_obs())
                image_observations.append(obs.get_Image_obs())
            # im.fromarray(np.hstack(obss[i].get_Image_obs()).astype(np.float32) * 255).show()
            if np.random.random() > epsilon:
                action_Qs = agent.act(np.array(vector_observations), np.array(image_observations))
                q_o_ts.append(np.array([np.max(action_Q) for action_Q in action_Qs]))
                action_ids = np.array([np.argmax(action_Q[0]) for action_Q in action_Qs], dtype= np.int8)
            else:
                action_ids = np.random.randint(n_actions, size = n_drones)

            for i, action_id in enumerate(action_ids):
                obss[i].action = action_id
            curr_obss_copy = copy.deepcopy(obss)
            new_obss, reward, _, _ = env.step(obss, action_ids)
            # print(reward)
            # To update relative observation of new_obs w.r.to other obs.
            for new_obs in new_obss:
                new_obs.UpdateObsFromOthers(unchanged_obs)
            old_vector_observations, old_image_observations, new_vector_observations, new_image_observations = [], [], [], []
            for i in range(n_drones):
                old_vector_observations.append(curr_obss_copy[i].get_Vector_obs())
                old_image_observations.append(curr_obss_copy[i].get_Image_obs())
                new_vector_observations.append(new_obss[i].get_Vector_obs())
                new_image_observations.append(new_obss[i].get_Image_obs())

                coll_coverage_ts += obss[i].get_Image_obs()[0]
                fire_map_coverage_indi[i] += obss[i].get_Image_obs()[0]
                fire_map_coverage += obss[i].get_Image_obs()[0]

            agent.step(np.array(old_vector_observations), np.array(old_image_observations), action_ids, reward, np.array(new_vector_observations), np.array(new_image_observations))
            # obss = new_obss # No need obss = new_obss, as the copy is by reference.

            if agent.check_train(frame_num):
                training_metrics = agent.learn()
                q_tr_o_ts.append([tr_metric.cpu().detach().numpy() for tr_metric in training_metrics[0]])
                q_tot_o_ts.append(training_metrics[1].cpu().detach().numpy())
                loss_o_ts.append(training_metrics[2])
            
            rew_o_ts.append(reward)
        
            Mt.append([(np.sum((fire_map - new_obs.image.belief_map).clip(0))) for new_obs in new_obss])
            coll_Mt.append((np.sum((fire_map - coll_coverage_ts.clip(max=1)).clip(0))))
            if (episode % 10 == 0) and (frame_num > e1):
                env.render(obss)

        policy_update_idx = np.argmax(np.array(rew_o_ts).sum(axis=0))
        agent.update_policy(policy_update_idx)

        Mt = (Mt / fire_map_count)
        total_Mt = np.sum(Mt, axis=0)
        coll_Mt = (coll_Mt / fire_map_count)
        coll_Sm = np.sum(coll_Mt)

        print('Episode ', episode, " FireMiss= {:.2f}".format(np.min(total_Mt) * 100), ' % Rew.Ov.Ep.= ', np.array(rew_o_ts).sum(axis=0), ' Coverage= {:.2f}%'.format((np.sum(fire_map_coverage.clip(max = 1))/len(env.universal_fire_set)) * 100))
        
        # if ((episode+1) % 50 == 0):
        #     original_fire_map = np.full([x_len, y_len], 0, dtype=np.int8)
        #     for env_x, env_y in env.universal_fire_set:
        #             original_fire_map[env_x][env_y] = 1
        #     with open(os.path.join( model_path, "figureData", "data_ep"+str(episode)), "wb") as f:
        #         pickle.dump([original_fire_map, fire_map_coverage.clip(max = 1), [fire_map_coverage_indi[i].clip(max = 1) for i in range(n_drones)], str(np.sum(fire_map_coverage.clip(max = 1))) +",\n"+ str(len(env.universal_fire_set)) +",\n"+ "{:.2f}%".format((np.sum(fire_map_coverage.clip(max = 1))/len(env.universal_fire_set)))], f)
        
        miss_history.append(total_Mt)
        coll_miss_history.append(coll_Sm)
        reward_history.append(np.array(rew_o_ts).sum(axis=0))
        coverage_history_indi.append([np.sum(fire_map_coverage_indi[i].clip(max = 1)) / len(env.universal_fire_set) for i in range(n_drones)])
        coverage_history.append(np.sum(fire_map_coverage.clip(max = 1)) / len(env.universal_fire_set))
        q_history.append([(sum(q_o_ts[i])/len(q_o_ts[i]) if len(q_o_ts[i])>0 else -1.0) for i in range(n_drones)])
        q_tr_history.append(np.mean(q_tr_o_ts, axis = 0))
        q_tot_history.append(np.mean(q_tot_o_ts, axis = 0))
        loss_history.append(np.mean(loss_o_ts, axis = 0))
                
        if (episode+1) % model_saving_step == 0 :
            agent.save_models()
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

    # end = time.time()
    # total_time = end-start
    # print("Total Time taken: ",total_time)
    # with open(os.path.join( model_path, "figureData", "tr_duration"), "wb") as Tp:
    #     pickle.dump(total_time, Tp)