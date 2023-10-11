# CRADLE: Cooperative Adaptive Decentralized Learning and Execution for UAV network to monitor Wildfire Front

This is the codification used in the ICRA 2024 paper proposing the CRADLE framework, an effective cooperative decentralized approach to monitor the wildfire front using multi-agent deep reinforcement learning. You are free to use all or part of the codes here presented for any purpose, provided that the paper is properly cited and the original authors properly credited. All the files here shared come with no warranties.
This project is built using Python v3.9.13 and PyTorch v1.13.0.dev20220930 over the system equipped with Apple M1 8 cores chip, Apple M1 graphics chipset and Memory 16 GB LPDDR4 configurations. Some .ipynb files are used to generate plots and images, so to access them, you need to install Jupyter Notebook (http://jupyter.readthedocs.io/en/latest/install.html) or VS Code with Jupyter notebook extension.


## Abstract

In this paper, we investigate the challenges of continuous wildfire front monitoring using Unmanned Aerial Vehicles (UAVs). While several UAV learning and execution strategies exist to effectively monitor the wildfire progression, they grapple with centralized limitations such as scalability, single-point-of-failure, lack of adaptability, etc. To address these, we introduce a novel decentralized framework, named Cooperative Adaptive Decentralized Learning and Execution (CRADLE), diverging from conventional Centralized Training and Decentralized Execution (CTDE) models. CRADLE promotes adaptive coordination among UAVs, leveraging a double Q-learning (DDQN) algorithm with a weighted experience aggregation mechanism. Our evaluations indicate that CRADLE not only outperforms state-of-the-art CTDE models by a margin of at least $7.763\%$ in terms of incurred loss but also exhibits a faster convergence rate.


## Files

This repository contains 2 folders: training_phase and testing_phase.

In training_phase folders, there are MSTA, VDN, and CRADLE sub-folders representing the baseline CTDE approaches: Multiple Single-Trained Agents (MSTA) and Value Decomposition Networks (VDN), and our proposed CRADLE approach, respectively, followed by the graph file to plot and compare the performance of the 3 apporaches.

Both CTDE approaches contain
1. main: performs the root operations.
2. environment: progresses fire and interacts with the UAV network.
3. agent: performs the deep reinforcement learning model.
4. model: the neural network.

The CRADLE approach contain
1. main: performs the root operations.
2. environment: progresses fire and interacts with the UAV network.
3. UAVagent: handles the interactions of a UAV with the environment and other UAVs.
4. NNagent: performs the deep reinforcement learning model.
5. model: the neural network.

The trained policies of all UAVs are stored in the models folder inside their respective approaches. To uniquely identify the trained policies, the policy folder name is interpreted with date & time, which is as follows: t2023-09-12 15_26_44.451624 ("t" followed by "YYYY-MM-DD HH_mm_ss.ssssss").

In testing_phase folders, the performance of the trained policies is evaluated in terms of scalability, and different environment settings.
The variation of environment settings are as follows,
1. Field Dimensions - 100x100, 100x150, 150x150
2. Fire Propagation Shape - Circular, Arc, T-Shape
3. Wind Condition - Windy, Non-windy

By default, the policies are trained in the "100x100, Circular and Non-windy" condition configured environment with 3 UAVs. The performance of the trained policies in the default setting over increased scalability (#UAVs: 3, 5, 7, 9, 11) is evaluated in the test_n sub-folder. The performance of various environment settings is evaluated in the further sub-folders with respect to the increasing number of UAVs. Note that, the variations are evaluated by changing the specific configuration, for example, if the evaluation is with respect to the Windy condition, then the setting is "100x100, Circular and Windy" and if the variation is with Arc shape, then the setting is "100x100, Arc and Non-windy". You are free to evaluate the performance of any set of configurations.

From this file sections, in summary, we have, 

there are 5 Python files to interpret,
1. /training_phase/MSTA/main.py
2. /training_phase/VDN/main.py
3. /training_phase/CRADLE/main.py
4. /testing_phase/test_n/main.py
5. /testing_phase/test_dim/dim150_150/main.py
6. /testing_phase/test_dim/dim100_150/main.py
7. /testing_phase/test_dim/dim50_50/main.py
8. /testing_phase/test_shape/arc/main.py
9. /testing_phase/test_shape/t_shape/main.py
10. /testing_phase/test_wind/main.py

3 Python notebooks to run,
1. /training_phase/graph.ipynb


## How to use <br />

# 1. To train MSTA/VDN/CRADLE approach:
In the torch environment terminal, from the root, change the directory to /training_phase/MSTA/ or /training_phase/VDN/ or /training_phase/CRADLE/ and type
    python main.py
After training overall episodes, the policies and plot data files are stored in /models folder under a particular timestamp, ex. "t2023-08-17 16:43:57.654315".

# 2. To train MSTA/VDN/CRADLE approach with various environment field dimensions:
In the folders, from the root, change the directory to /training_phase/MSTA/ or /training_phase/VDN/ or /training_phase/CRADLE/ and open main.py.
Change the x_len and y_len parameters. By default they are x_len = 100 and y_len = 100.
Save the file and in the torch environment terminal, type
    python main.py
After training overall episodes, the policies and plot data files are stored in /models folder under a particular timestamp, ex. "t2023-08-17 16:43:57.654315".

# 3. To plot the performance of the training in terms of avg. rewards earned, team coverage, avg. training loss and avg. convergence:
In directory /training_phase/, open graph.ipynb in jupyter notebook or VS Code.
Under 2.1. File Path section, change the directory path of all 3 approaches to the trained policies (Most recently trained).
Run all the code inside the file to generate the 3 performance comparison graphs in terms of avg. rewards earned and team coverage followed by avg. training loss and avg. convergence plots.

# 4. To test the CRADLE approach:
The performance of the CRADLE approach can be evaluated in terms of various environmental conditions as mentioned before. 
To test the trained policies in any environment, copy the policy folders from /training_phase/CRADLE/models folder, change the directory to the environment of your choice like /testing_phase/test_shape/t_shape/ and paste the policy folder in /models folder. The policy folder name would look like "t2023-08-17 16:43:57.654315".
Then change the current directory back to the chosen environment (e.g. /testing_phase/test_shape/t_shape/).
In main.py, replace load_path = "" with the policy folder name that you plan to test like load_path = "t2023-08-17 16:43:57.654315" and save the file.
In the torch environment terminal, type
    python main.py
After testing each set of UAVs over 10 episodes, the coverage and time taken for the team to reach 80% coverage is saved in /data/uav_# folder in terms of UAVs count. The final test reports avgd_coverage_o_n.csv and coverage_time_o_n.csv contains the averaged values over each set of UAVs (3, 5, 7, 9, 11).

## Contact
For questions about the codification or paper, <br />please send an email to gauravsrikar@nevada.unr.edu, mdtamjidh@nevada.unr.edu or aralab2018@gmail.com.

