# Author: Victor Augusto Kich
# Github: https://github.com/victorkich
# E-mail: victorkich@yahoo.com.br

import numpy as np
import tensorflow as tf
from datetime import datetime
import environment
import threading

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
            obs2, rew, done, _ = [1,3,4]


if __name__ == '__main__':
    env = environment.arm()
    env.start()
    environment.showPlot(env)

    '''
    PPO(hidden_sizes=[64,64], cr_lr=5e-4, ac_lr=2e-4, gamma=0.99, lam=0.95,\
        steps_per_env=5000, number_envs=1, eps=0.15, actor_iter=6,\
        critic_iter=10, action_type='Box', num_epochs=5000, minibatch_size=256)
    '''
