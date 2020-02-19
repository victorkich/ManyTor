# Author: Victor Augusto Kich
# Github: https://github.com/victorkich
# E-mail: victorkich@yahoo.com.br

from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib.animation import FuncAnimation
import threading
import numpy as np
import pandas as pd
import time
import random
import math
import fkmath as fkm
from tqdm import tqdm

np.set_printoptions(precision=4, linewidth=160)
fig = plt.figure()

def showPlot(arm):
    ani = FuncAnimation(fig, arm.animate, interval=1)
    plt.show()

class arm(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.zeros = np.array([210.0, 180.0, 65.0, 153.0])
        self.goals = np.array([0.0 for i in range(4)])
        self.plotpoints = False
        self.trajectory = pd.DataFrame({'x':[], 'y':[], 'z':[]})
        self.obj_number = 0
        self.obj_remaining = 0
        self.ax = plt.gca(projection='3d')
        self.reset = False
        self.done = False
        self.n_obs = [0,0,0,0]
        self.actual_epoch = 0
        self.actual_step = 0
        self.old_fixed_reward = 0

    def run(self):
        rt = threading.Thread(name = 'realtime', target = self.realtime)
        rt.setDaemon(True)
        rt.start()
        time.sleep(1.0)

        obj = threading.Thread(name = 'objectives', target = self.objectives)
        obj.setDaemon(True)
        obj.start()
        time.sleep(1.0)

        dis = threading.Thread(name = 'distance', target = self.distanced)
        dis.setDaemon(True)
        dis.start()
        time.sleep(1.0)

    def step(self, act, actual_epoch, actual_step, normalize):
        self.actual_epoch = actual_epoch
        self.actual_step = actual_step
        self.old_distance = self.distance
        if normalize:
            goals = np.array([fkm.angleNormalize(act[i]) for i in range(4)])
        else:
            goals = np.array([act[i] for i in range(4)])
        self.ctarget(goals, 100)
        rew = self.get_episode_reward()
        obs2 = self.distance
        return obs2, rew, self.done

    def get_episode_reward(self):
        fixed_reward = (self.obj_number-self.obj_remaining)/self.obj_number-self.old_fixed_reward
        threshold = 100
        touch_ground = 0
        if self.negative_reward:
            touch_ground = -200
        weight = self.obj_number/threshold
        variable_reward = []
        for i in range(int(self.obj_number)):
            if((self.old_distance[i] <= threshold) and self.old_distance[i] != 0):
                variable_reward.append(weight/self.old_distance[i])
        old_variable_reward = sum(variable_reward)
        for i in range(int(self.obj_number)):
            if((self.distance[i] <= threshold) and self.distance[i] != 0):
                variable_reward.append(weight/self.distance[i])
        variable_reward = sum(variable_reward)
        reward = ((fixed_reward*2 + (variable_reward - old_variable_reward)) * 100) + touch_ground
        self.old_fixed_reward = fixed_reward
        if self.negative_reward:
            _ = self.resety()
        return reward

    def get_episode_length(self):
        return self.obj_number

    def resety(self):
        self.reset = True
        self.old_fixed_reward = 0
        self.negative_reward = False
        self.goals = np.array([0.0 for i in range(4)])
        return self.distance

    def clear_tratectory(self):
        self.trajectory = pd.DataFrame({'x':[], 'y':[], 'z':[]})

    def animate(self, i):
        x, y, z = [np.array(i) for i in [self.df.x, self.df.y, self.df.z]]
        self.ax.clear()
        self.ax.plot3D(x, y, z, 'gray', label='Links', linewidth=5)
        self.ax.scatter3D(x, y, z, color='black', label='Joints')
        self.ax.scatter3D(x[3], y[3], zs=0, zdir='z', label='Projection', color='red')
        #self.ax.scatter3D(0, 0, 4.3, plotnonfinite=False, s=135000, norm=1, alpha=0.2, lw=0)
        x, y, z = [np.array(i) for i in [self.trajectory.x, self.trajectory.y, self.trajectory.z]]
        self.ax.plot3D(x, y, z, c='b', label='Trajectory')

        if self.plotpoints == True:
            x, y, z, = [], [], []
            n_x, n_y, n_z = [np.array(i) for i in [self.points.x, self.points.y, self.points.z]]
            for i in range(self.obj_number[0]):
                if self.boolplot[i]:
                    x.append(n_x[i])
                    y.append(n_y[i])
                    z.append(n_z[i])

            legend = 'Objectives: ' + str(self.obj_remaining[0]) + '/' + str(self.obj_number[0])
            self.ax.scatter3D(x, y, z, color='green', label=legend)

        title = 'Epoch: ' + str(self.actual_epoch) + ' Step: ' + str(self.actual_step)
        self.ax.set_title(title, size=10)
        self.ax.legend(loc=2, prop={'size':7})
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')
        self.ax.set_zlabel('z')
        self.ax.set_xlim([-60, 60])
        self.ax.set_ylim([-60, 60])
        self.ax.set_zlim([0, 60])

    def objectives(self):
        while True:
            self.done = False
            self.reset = False
            self.obj_number = np.array([10]) #np.random.randint(low=10, high=11, size=1)
            self.obj_remaining = np.array([10])
            self.points = []
            self.points.append([51.3, 0, 0])
            cont = 0
            self.points.append([0,51,0])
            while cont < self.obj_number:
                rands = [random.uniform(-51.3, 51.3) for i in range(3)]
                if rands[2] >= 0:
                    value = math.sqrt(math.sqrt(rands[0]**2 + rands[1]**2)**2 + rands[2]**2)
                    if value <= 51.3:
                        self.points.append(rands)
                        cont = cont + 1
            self.points = pd.DataFrame(self.points)
            self.points.rename(columns = {0:'x', 1:'y', 2:'z'}, inplace=True)
            self.plotpoints = True
            self.boolplot = [True for i in range(10)]
            time.sleep(2)
            while not (self.done or self.reset):
                for p in range(int(self.obj_number)):
                    validation_test = []
                    for a in range(3):
                        if(math.isclose(self.df.iat[3, a], self.points.iat[p, a],\
                                        abs_tol=0.75)):
                            validation_test.append(True)
                        else:
                            validation_test.append(False)
                    if all(validation_test):
                        self.points.iloc[p] = 0.0
                        self.boolplot[p] = False
                        self.obj_remaining -= 1
                if self.distance.all() == 0.0:
                    self.done = True
                time.sleep(0.1)
            while self.reset == False:
                time.sleep(0.5)

    def realtime(self):
        while True:
            # Modes -> 1 = first joint / 2 = second joint
            #          3 = third joint / 4 = fourth joint
            df = pd.DataFrame(np.zeros(3)).T
            df2 = pd.DataFrame(fkm.fk(mode=i, goals=self.goals)[0:3, 3] for i in range(2,5))
            df = df.append(df2).reset_index(drop=True)
            df.rename(columns = {0:'x', 1:'y', 2:'z'}, inplace=True)
            self.df = pd.DataFrame(df)
            self.trajectory = self.trajectory.append(self.df.iloc[3])
            self.trajectory.drop_duplicates(inplace=True)
            if self.df.iloc[3, 2] < 0:
                self.negative_reward = True
            time.sleep(0.1)

    def sample(self):
        sample = [np.squeeze(np.random.randint(low=-120, high=121, size=1)[0]) for i in range(4)]
        return sample

    def distanced(self):
        while True:
            distance = pd.DataFrame({'obj_dist':[]})
            for p in range(int(self.obj_number)):
                x, y, z = [(abs(self.df.iloc[3, i] - self.points.iloc[p, i])) for i in range(3)]
                dist = pd.DataFrame({'obj_dist':[math.sqrt(math.sqrt(x**2 + y**2)**2 + z**2)]})
                distance = distance.append(dist).reset_index(drop=True)
            self.distance = np.squeeze(distance.T.values)
            time.sleep(0.1)

    def ctarget(self, targ, iterations):
        self.stop = False
        dtf = threading.Thread(name = 'dataflow',target = self.dataFlow,\
                               args = (targ, iterations, ))
        dtf.setDaemon(True)
        dtf.start()
        while True:
            if self.stop == True:
                break
            time.sleep(0.05)

    def dataFlow(self, targ, iterations):
        track = np.linspace(self.goals, targ, num=iterations)
        for t in tqdm(range(len(track))):
            self.goals = track[t]
            time.sleep(0.075)
        #print(track[len(track)-1])
        self.stop = True
