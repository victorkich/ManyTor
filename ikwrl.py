# Author: Victor Augusto Kich
# Github: https://github.com/victorkich
# E-mail: victorkich@yahoo.com.br

import math
import numpy as np
import pandas as pd
import time
import threading
import random
import tensorflow as tf
from datetime import datetime

def mlp(x, hidden_layers, output_layer, activation=tf.tanh, last_activation=None):
    ''' Multi-layer perceptron
    '''
    for l in hidden_layers:
        x = tf.layers.dense(x, units=l, activation=activation)
    return tf.layers.dense(x, units=output_layer, activation=last_activation)

def softmax_entropy(logits):
    ''' Softmax Entropy
    '''
    return -tf.reduce_sum(tf.nn.softmax(logits, axis=-1) * tf.nn.log_softmax(logits, axis=-1), axis=-1)

def clipped_surrogate_obj(new_p, old_p, adv, eps):
    ''' Clipped surrogate objective function
    '''
    rt = tf.exp(new_p - old_p) # i.e. pi / old_pi
    return -tf.reduce_mean(tf.minimum(rt*adv, tf.clip_by_value(rt, 1-eps, 1+eps)*adv))

def GAE(rews, v, v_last, gamma=0.99, lam=0.95):
    ''' Generalized Advantage Estimation
    '''
    assert len(rews) == len(v)
    vs = np.append(v, v_last)
    delta = np.array(rews) + gamma*vs[1:] - vs[:-1]
    gae_advantage = discounted_rewards(delta, 0, gamma*lam)
    return gae_advantage

def discounted_rewards(rews, last_sv, gamma):
    ''' Discounted reward to go
        Parameters:
        ----------
        rews: list of rewards
        last_sv: value of the last state
        gamma: discount value
    '''
    rtg = np.zeros_like(rews, dtype=np.float32)
    rtg[-1] = rews[-1] + gamma*last_sv
    for i in reversed(range(len(rews)-1)):
        rtg[i] = rews[i] + gamma*rtg[i+1]
    return rtg

def gaussian_log_likelihood(x, mean, log_std):
    ''' Gaussian Log Likelihood
    '''
    log_p = -0.5 *((x-mean)**2 / (tf.exp(log_std)**2+1e-9) + 2*log_std + np.log(2*np.pi))
    return tf.reduce_sum(log_p, axis=-1)

class Buffer():
    ''' Class to store the experience from a unique policy
    '''
    def __init__(self, gamma=0.99, lam=0.95):
        self.gamma = gamma
        self.lam = lam
        self.adv = []
        self.ob = []
        self.ac = []
        self.rtg = []

    def store(self, temp_traj, last_sv):
        ''' Add temp_traj values to the buffers and compute the advantage and reward to go
            Parameters:
            -----------
            temp_traj: list where each element is a list that contains:
                       observation, reward, action, state-value
            last_sv: value of the last state (Used to Bootstrap)
        '''
        # store only if there are temporary trajectories
        if len(temp_traj) > 0:
            self.ob.extend(temp_traj[:,0])
            rtg = discounted_rewards(temp_traj[:,1], last_sv, self.gamma)
            self.adv.extend(GAE(temp_traj[:,1], temp_traj[:,3], last_sv, self.gamma, self.lam))
            self.rtg.extend(rtg)
            self.ac.extend(temp_traj[:,2])

    def get_batch(self):
        # standardize the advantage values
        norm_adv = (self.adv - np.mean(self.adv)) / (np.std(self.adv) + 1e-10)
        return np.array(self.ob), np.array(self.ac), np.array(norm_adv), np.array(self.rtg)

    def __len__(self):
        assert(len(self.adv) == len(self.ob) == len(self.ac) == len(self.rtg))
        return len(self.ob)

def PPO(hidden_sizes=[32], cr_lr=5e-3, ac_lr=5e-3, num_epochs=50, \
        minibatch_size=5000, gamma=0.99, lam=0.95, number_envs=1, eps=0.1, \
        actor_iter=5, critic_iter=10, steps_per_env=100, action_type='Discrete'):

    # Placeholders
    act_dim = 1
    obs_dim = 1
    act_ph = tf.placeholder(shape=(None, act_dim), type=tf.float32, name='act')
    obs_ph = tf.placeholder(shape=(None, obs_dim), dtype=tf.float32, name='obs')
    ret_ph = tf.placeholder(shape=(None,), dtype=tf.float32, name='ret')
    adv_ph = tf.placeholder(shape=(None,), dtype=tf.float32, name='adv')
    old_p_log_ph = tf.placeholder(shape=(None,), dtype=tf.float32, name='old_p_log')

    with tf.variable_scope('actor_nn'):
        p_logits = mlp(obs_ph, hidden_sizes, act_dim, tf.nn.relu, last_activation=tf.tanh)

    act_smp = tf.squeeze(tf.random.multinomial(p_logits, 1))
    act_onehot = tf.one_hot(act_ph, depth=act_dim)
    p_log = tf.reduce_sum(act_onehot * tf.nn.log_softmax(p_logits), axis=-1)

    with tf.variable_score('critic_nn'):
        s_values = mlp(obs_ph, hidden_sizes, 1, tf.tanh, last_activation=None)
        s_values = tf.squeeze(s_values)

    # PPO loss function
    p_loss = clipped_surrogate_obj(p_log, old_p_log_ph, adv_ph, eps)
    # MSE loss function
    v_loss = tf.reduce_mean((ret_ph - s_values)**2)

    # Policy optimizer
    p_opt = tf.train.AdamOptimizer(ac_lr).minimize(p_loss)
    # Value function optimizer
    v_opt = tf.train.AdamOptimizer(cr_lr).minimize(v_loss)

    # Time
    now = datetime.now()
    clock_time = "{}_{}.{}.{}".format(now.day, now.hour, now.minute, now.second)
    print('Time: ', clock_time)

    # Create a session
    sess = tf.Session()
    # Initialize the variables
    sess.run(tf.global_variables_initializer())

    # Variable to store the total number of steps
    step_count = 0

    print('Env batch size: ', steps_per_env, ' Batch size: ', steps_per_env*number_envs)

    for ep in range(num_epochs):
        # Create the buffer that will contain the trajectories (full or partial)
        # Run with the last policy
        buffer = Buffer(gamma, lam)
        # Lists to store rewards and length of the trajectories completed
        batch_rew = []
        batch_len = []
        temp_buf = []

        for _ in range(steps_per_env):
            # Iterate over a fixed number of steps
            act, val = sess.run([act_smp, s_values], feed_dict={obs:ph[1]})
            act = np.squeeze(act)

            # Take a step in the environment
            obs2, rew, done, _ =


#-------------------------------------------------------------------------------

def deg2rad(deg):
    ''' Convert angles from degress to radians
    '''
    return np.pi * deg / 180.0

def rad2deg(rad):
    ''' Converts angles from radians to degress
    '''
    return 180.0 * rad / np.pi

def dh(a, alfa, d, theta):
    ''' Builds the Homogeneous Transformation matrix
        corresponding to each line of the Denavit-Hartenberg
        parameters
    '''
    m = np.array([
        [np.cos(theta), -np.sin(theta)*np.cos(alfa),
        np.sin(theta)*np.sin(alfa), a*np.cos(theta)],
        [np.sin(theta), np.cos(theta)*np.cos(alfa),
        -np.cos(theta)*np.sin(alfa), a*np.sin(theta)],
        [0, np.sin(alfa), np.cos(alfa), d],
        [0, 0, 0, 1]
    ])
    return m

class ArmRL(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)

        self.zeros = np.array([210.0, 180.0, 65.0, 153.0])
        self.goals = np.array([0.0 for i in range(4)])
        self.plotpoints = False
        self.trajectory = pd.DataFrame({'x':[], 'y':[], 'z':[]})
        self.obj_number = 0

    def run(self):
        rt = threading.Thread(name = 'realtime', target = self.realtime)
        rt.setDaemon(True)
        rt.start()

        obj = threading.Thread(name = 'objectives', target = self.objectives)
        obj.setDaemon(True)
        obj.start()

        #goals = np.array([-50, 50, 150, -60])
        #self.ctarget(goals, 250)
        '''
        PPO(hidden_sizes=[64,64], cr_lr=5e-4, ac_lr=2e-4, gamma=0.99, lam=0.95,\
            steps_per_env=5000, number_envs=1, eps=0.15, actor_iter=6,\
            critic_iter=10, action_type='Box', num_epochs=5000, minibatch_size=256)
        '''

    def objectives(self):
        while True:
            self.obj_number = np.random.randint(low=5, high=25, size=1)
            self.points = []
            self.points.append([51.3, 0, 0])
            cont = 0
            while cont < self.obj_number:
                rands = [random.uniform(-51.3, 51.3) for i in range(3)]
                if rands[2] >= 0:
                    value = math.sqrt(math.sqrt(rands[0]**2 + rands[1]**2)**2 + rands[2]**2)
                    if value <= 51.3:
                        self.points.append(rands)
                        cont = cont + 1
            self.points = pd.DataFrame(self.points)
            self.points.rename(columns = {0:'x', 1:'y', 2:'z'}, inplace=True)
            print(self.points)
            self.plotpoints = True
            while True:
                for p in range(int(self.obj_number)):
                    validation_test = []
                    for a in range(3):
                        if(math.isclose(self.df.iat[3, a], self.points.iat[p, a],\
                                        abs_tol=0.5)):
                            validation_test.append(True)
                        else:
                            validation_test.append(False)
                    if all(validation_test):
                        self.points.drop(p, inplace=True)
                time.sleep(0.01)

    def fk(self, mode):
        ''' Forward Kinematics
        '''
        # Convert angles from degress to radians
        t = [deg2rad(x) for x in self.goals]
        # Register the DH parameters
        hs = []
        hs.append(dh(0,       -np.pi/2, 4.3,  t[0]))
        if mode >= 2:
            hs.append(dh(0,    np.pi/2, 0.0,  t[1]))
        if mode >= 3:
            hs.append(dh(0,   -np.pi/2, 24.3, t[2]))
        if mode == 4:
            hs.append(dh(27.0, np.pi/2, 0.0,  t[3] - np.pi/2))

        m = np.eye(4)
        for h in hs:
            m = m.dot(h)
        return m

    def realtime(self):
        while True:
            # Modes -> 1 = first joint / 2 = second joint
            #          3 = third joint / 4 = fourth joint
            df = pd.DataFrame(np.zeros(3)).T
            df2 = pd.DataFrame(self.fk(mode=i)[0:3, 3] for i in range(2,5))
            df = df.append(df2).reset_index(drop=True)
            df.rename(columns = {0:'x', 1:'y', 2:'z'}, inplace=True)
            self.df = df
            self.trajectory = self.trajectory.append(self.df.iloc[3])
            self.trajectory.drop_duplicates(inplace=True)

            distance = pd.DataFrame({'obj_dist':[]})
            for p in range(int(self.obj_number)):
                x, y, z = [(abs(self.df.iloc[3, i] - self.points.iloc[p, i]))\
                           for i in range(3)]
                dist = pd.DataFrame({'obj_dist':[math.sqrt(math.sqrt(x**2 + y**2)**2 + z**2)]})
                distance = distance.append(dist).reset_index(drop=True)
            print(distance)
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
            time.sleep(0.1)

    def dataFlow(self, targ, iterations):
        track = np.linspace(self.goals, targ, num=iterations)
        for t in track:
            self.goals = t
            time.sleep(0.1)
        self.stop = True

arm = ArmRL()
fig = plt.figure()
ax = plt.gca(projection='3d')

def animate(i):
    x, y, z = [np.array(i) for i in [arm.df.x, arm.df.y, arm.df.z]]
    ax.clear()
    ax.plot3D(x, y, z, 'gray', label='Links', linewidth=5)
    ax.scatter3D(x, y, z, color='black', label='Joints')
    ax.scatter(x[3], y[3], zs=0, zdir='z', label='Projection', color='red')
    ax.scatter3D(0, 0, 4.3, plotnonfinite=False, s=135000, norm=1, alpha=0.2, lw=0)
    x, y, z = [np.array(i) for i in [arm.trajectory.x, arm.trajectory.y, arm.trajectory.z]]
    ax.plot3D(x, y, z, c='b', label='Trajectory')

    if arm.plotpoints == True:
        x, y, z = [np.array(i) for i in [arm.points.x, arm.points.y, arm.points.z]]
        ax.scatter3D(x, y, z, color='green', label='Objectives')

    ax.legend(loc=2, prop={'size':10})
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_xlim([-60, 60])
    ax.set_ylim([-60, 60])
    ax.set_zlim([0, 60])

ani = FuncAnimation(fig, animate, interval=1)
arm.start()
plt.show()
