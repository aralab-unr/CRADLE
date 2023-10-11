import numpy as np
import gymnasium as gym
import cv2 as cv
import sys

# np.set_printoptions(threshold=sys.maxsize)

class FireEnvironment(gym.Env):
  
    def __init__ (self, x_len, y_len):
        super().__init__()

        self.x_len = x_len
        self.y_len = y_len
        self.fire_x_start = 48
        self.fire_y_start = 48
        self.fire_x_end = 52
        self.fire_y_end = 52
        self.flame_min = 15
        self.flame_max = 20

        self.dmax = 2.5
        self.K = 0.05
        self.d_sight = 10
        self.fire_count = 0

        self.universal_fire_set = set()

    def reset(self, uavs):
        self.binary_val = np.full([self.x_len, self.y_len], 0, dtype=np.int8)
        self.fuel_map = np.random.randint(self.flame_min, self.flame_max+1, [self.x_len, self.y_len])
        self.fire_set = set()
        self.fire_off = set()
        self.universal_fire_set = set()

        for i in range(self.fire_x_start, self.fire_x_end+1):
            for j in range(self.fire_y_start, self.fire_y_end+1):
                self.binary_val[i][j] = 1
                self.fire_set.add((i,j))
                self.universal_fire_set.add((i,j))

        self.fire_count = len(self.universal_fire_set)

        for uav in uavs:
            uav.obs.image.belief_map = self.binary_val.copy()

        self.env_frames = []
        
    def updateObsAfterReset(self, uavs):
        for uav in uavs:
            _ = self.reward(uav)

    def simStep(self):
        fire_neighbor = set()
        fire_off = set()
        for i, j in self.fire_set:
            x_nei = np.arange(i-2,i+2+1)
            x_nei = x_nei[(x_nei>=0) & (x_nei<self.x_len)]
            y_nei = np.arange(j-2, j+2+1)
            y_nei = y_nei[(y_nei>=0) & (y_nei<self.y_len)]
            x_nei_coords, y_nei_coords = np.meshgrid(x_nei, y_nei)
            for k, l in zip(x_nei_coords.flatten(), y_nei_coords.flatten()):
                if (k,l) not in self.fire_set:
                    if self.fuel_map[i][j] > 0:
                        fire_neighbor.add((k,l))
            if self.fuel_map[i][j] > 0:
                self.fuel_map[i][j] -= 1
            else:
                fire_off.add((i,j))
        
        #burn_count = 0
        #total_count = 0
        pre_univ_fire_count = len(self.universal_fire_set)
        new_fire_set = set()
        for i, j in fire_neighbor:
            x_nei = np.arange(i-2,i+2+1)
            x_nei = x_nei[(x_nei>=0) & (x_nei<self.x_len)]
            y_nei = np.arange(j-2, j+2+1)
            y_nei = y_nei[(y_nei>=0) & (y_nei<self.y_len)]
            x_nei_coords, y_nei_coords = np.meshgrid(x_nei, y_nei)

            Pnmkl = 1
            pres_nei = set()
            for k, l in zip(x_nei_coords.flatten(), y_nei_coords.flatten()):
                if (k,l) in self.fire_set:
                    d_nmkl = np.linalg.norm(np.array([i,j]) - np.array([k,l]))
                    if d_nmkl < self.dmax:
                        Pnmkl *= (1 - max(0, min(1, ((self.K * (1/(d_nmkl**2)))))))
            unif_comp = np.random.uniform(0,1,1)
            # Stochastically ignite fire
            if ((1 - Pnmkl)>=unif_comp)[0]:
                new_fire_set.add((i,j))
                self.universal_fire_set.add((i,j))
                #burn_count+=1
            #total_count+=1

        for new_fire in new_fire_set:
            self.fire_set.add(new_fire)
            self.binary_val[new_fire[0]][new_fire[1]] = 1
        
        #if total_count==0 : total_count = 1
        #print(np.round((burn_count/total_count)*100),"% (",burn_count," / ",total_count,")")
            
        for ind_fire_off in fire_off:
            self.fire_off.add(ind_fire_off)
            self.fire_set.remove(ind_fire_off)
            self.binary_val[ind_fire_off[0]][ind_fire_off[1]] = 0

        self.fire_count = len(self.universal_fire_set) - pre_univ_fire_count

    def get_fire_map(self):
        return self.binary_val
    
    def observe(self, x_view, y_view):
        x_view_min = x_view - self.d_sight
        x_view_max = x_view + self.d_sight
        y_view_min = y_view - self.d_sight
        y_view_max = y_view + self.d_sight

        x_view_min = int( self.correct_coords(x_view_min, self.x_len) )
        x_view_max = int( self.correct_coords(x_view_max, self.x_len) )
        y_view_min = int( self.correct_coords(y_view_min, self.y_len) )
        y_view_max = int( self.correct_coords(y_view_max, self.y_len) )

        observed_view = self.binary_val[x_view_min:x_view_max, y_view_min:y_view_max]

        return observed_view

        # x_obs = np.arange(x_view - self.d_sight, x_view + self.d_sight +1)
        # x_obs = x_obs[(x_obs >= 0) & (x_obs < self.x_len)]
        # y_obs = np.arange(y_view - self.d_sight, y_view + self.d_sight + 1)
        # y_obs = y_obs[(y_obs >= 0) & (y_obs < self.y_len)]

        # for xi, yi in zip(x_obs.flatten(), y_obs.flatten()):
            
    def correct_coords(self, value, upper_limit):
        if value <= 0: 
            value = 0
        elif value >= upper_limit: 
            value = upper_limit - 1
        
        return value

    def step(self, uav, action):

        uav.obs.image.coverage_map[uav.obs.image.coverage_map > 0] -= 1

        if action == 0:
            uav.obs.vector.pos.x -= 1
        elif action == 1:
            uav.obs.vector.pos.x += 1
        elif action == 2:
            uav.obs.vector.pos.y += 1
        elif action == 3:
            uav.obs.vector.pos.y -= 1

        uav.obs.vector.pos.x = self.correct_coords(uav.obs.vector.pos.x, self.x_len)
        uav.obs.vector.pos.y = self.correct_coords(uav.obs.vector.pos.y, self.y_len)

        rew = self.reward(uav)

        return uav, rew, False, ""

    def render(self, uavs):
        xs = []
        ys = []
        for uav in uavs:
            xs.append(int(uav.obs.vector.pos.x))
            ys.append(int(uav.obs.vector.pos.y))
        self.print_env(xs, ys)

    def print_env(self, xs, ys):

        # Create a black image
        img = np.zeros((self.x_len, self.y_len, 3), np.uint8)

        # Make the image green
        cv.rectangle(img, (0,0), (self.y_len, self.x_len), (34,139,34), thickness = -1)

        # Print fire
        for x, y in self.fire_set:
            cv.line(img,(y, x),(y, x),(5,94,255),1)

        # Print burnt forest
        for x, y in self.fire_off:
            cv.line(img,(y, x),(y, x),(25, 25, 25),1)

        # Position of UAV
        for x, y in zip(xs, ys):
            cv.line(img,(y, x),(y, x),(238,207,112),1)

            # Min & Max of the view
            x_view_min = x-1 - self.d_sight
            x_view_max = x-1 + self.d_sight
            y_view_min = y-1 - self.d_sight
            y_view_max = y-1 + self.d_sight

            cv.rectangle(img, (y_view_min, x_view_min), (y_view_max+1, x_view_max+1), (0,255,255), thickness = 1)

        self.env_frames.append(img.copy()[:, :, ::-1])

        display = cv.resize(img, (self.y_len*5, self.x_len*5)) 
        cv.imshow("filled", display)
        cv.waitKey(1)

    def reward(self, uav):
        x_view = int(uav.obs.vector.pos.x)
        y_view = int(uav.obs.vector.pos.y)

        x_view_min = x_view - self.d_sight
        x_view_max = x_view + self.d_sight
        y_view_min = y_view - self.d_sight
        y_view_max = y_view + self.d_sight

        x_view_min = self.correct_coords(x_view_min, self.x_len)
        x_view_max = self.correct_coords(x_view_max, self.x_len)
        y_view_min = self.correct_coords(y_view_min, self.y_len)
        y_view_max = self.correct_coords(y_view_max, self.y_len)

        reward = np.sum((self.binary_val[x_view_min:x_view_max, y_view_min:y_view_max] - uav.obs.image.belief_map[x_view_min:x_view_max, y_view_min:y_view_max]).clip(min = 0))

        # print(uav.id ,reward, "[", np.sum(self.binary_val[x_view_min:x_view_max, y_view_min:y_view_max]), "/", np.sum(self.binary_val), "] [", np.sum(uav.obs.image.belief_map[x_view_min:x_view_max, y_view_min:y_view_max]), "/", np.sum(uav.obs.image.belief_map), "]", np.sum((self.binary_val[x_view_min:x_view_max, y_view_min:y_view_max] - uav.obs.image.belief_map[x_view_min:x_view_max, y_view_min:y_view_max]) < 0), len(self.fire_off))

        uav.obs.image.belief_map[x_view_min:x_view_max, y_view_min:y_view_max] = self.binary_val[x_view_min:x_view_max, y_view_min:y_view_max]
        uav.obs.image.coverage_map[x_view_min:x_view_max, y_view_min:y_view_max] = np.full(((x_view_max-x_view_min), (y_view_max-y_view_min)), 255, dtype=np.uint8)


        if reward < 0.0:
            reward = 0.0

        return reward