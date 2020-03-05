# Author: Victor Augusto Kich
# Github: https://github.com/victorkich
# E-mail: victorkich@yahoo.com.br

import numpy as np
import tensorflow as tf
from datetime import datetime
import environment
import threading
import fkmath as fkm
from collections import deque
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument("--train", help="Set the name of train model file", type=str)
parser.add_argument("--test", help="Select the train model file", type=str)
args = parser.parse_args()

tf.compat.v1.disable_eager_execution()

def mlp(x, hidden_layers, output_layer, activation=tf.nn.relu, last_activation=None):
    '''
    Multi-layer perceptron
    '''
    for l in hidden_layers:
        x = tf.compat.v1.layers.dense(x, units=l, activation=activation)
    return tf.compat.v1.layers.dense(x, units=output_layer, activation=last_activation)

def deterministic_actor_critic(x, a, hidden_sizes, act_dim, max_act):
    '''
    Deterministic Actor-Critic
    '''
    # Actor
    with tf.compat.v1.variable_scope('p_mlp'):
        p_means = max_act * mlp(x, hidden_sizes, act_dim, last_activation=tf.tanh)

    # Critic with as input the deterministic action of the actor
    with tf.compat.v1.variable_scope('q_mlp'):
        q_d = mlp(tf.concat([x,p_means], axis=-1), hidden_sizes, 1, last_activation=None)

    # Critic with as input an arbirtary action
    with tf.compat.v1.variable_scope('q_mlp', reuse=True): # Use the weights of the mlp just defined
        q_a = mlp(tf.concat([x,a], axis=-1), hidden_sizes, 1, last_activation=None)

    return p_means, tf.squeeze(q_d), tf.squeeze(q_a)

class ExperiencedBuffer():
    '''
    Experienced buffer
    '''
    def __init__(self, buffer_size):
        # Contains up to 'buffer_size' experience
        self.obs_buf = deque(maxlen=buffer_size)
        self.rew_buf = deque(maxlen=buffer_size)
        self.act_buf = deque(maxlen=buffer_size)
        self.obs2_buf = deque(maxlen=buffer_size)
        self.done_buf = deque(maxlen=buffer_size)

    def add(self, obs, rew, act, obs2, done):
        '''
        Add a new transition to the buffers
        '''
        self.obs_buf.append(obs)
        self.rew_buf.append(rew)
        self.act_buf.append(act)
        self.obs2_buf.append(obs2)
        self.done_buf.append(done)

    def sample_minibatch(self, batch_size):
        '''
        Sample a mini-batch of size 'batch_size'
        '''
        mb_indices = np.random.randint(len(self.obs_buf), size=batch_size)

        mb_obs = [self.obs_buf[i] for i in mb_indices]
        mb_rew = [self.rew_buf[i] for i in mb_indices]
        mb_act = [self.act_buf[i] for i in mb_indices]
        mb_obs2 = [self.obs2_buf[i] for i in mb_indices]
        mb_done = [self.done_buf[i] for i in mb_indices]

        return mb_obs, mb_rew, mb_act, mb_obs2, mb_done

    def __len__(self):
        return len(self.obs_buf)

def DDPG(env, hidden_sizes=[32], ac_lr=1e-2, cr_lr=1e-2, num_epochs=2000, buffer_size=5000, discount=0.99,
        batch_size=128, min_buffer_size=5000, tau=0.005):

    obs_dim = (10,)
    act_dim = (4,)

    # Create some placeholders
    obs_ph = tf.keras.backend.placeholder(shape=(None, obs_dim[0]), dtype=tf.float32, name='obs')
    act_ph = tf.keras.backend.placeholder(shape=(None, act_dim[0]), dtype=tf.float32, name='act')
    y_ph = tf.keras.backend.placeholder(shape=(None,), dtype=tf.float32, name='y')

    # Create an online deterministic actor-critic
    with tf.compat.v1.variable_scope('online'):
        p_onl, qd_onl, qa_onl = deterministic_actor_critic(obs_ph, act_ph, hidden_sizes, act_dim[0], 160)
    # and a target one
    with tf.compat.v1.variable_scope('target'):
        ########## act_dim[0]
        _, qd_tar, _ = deterministic_actor_critic(obs_ph, act_ph, hidden_sizes, act_dim[0], 160)

    def variables_in_scope(scope):
        '''
        Retrieve all the variables in the scope 'scope'
        '''
        return tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope)

    # Copy all the online variables to the target networks i.e. target = online
    # Needed only at the beginning
    init_target = [target_var.assign(online_var) for target_var, online_var in zip(variables_in_scope('target'), variables_in_scope('online'))]
    init_target_op = tf.group(*init_target)

    # Soft update
    update_target = [target_var.assign(tau*online_var + (1-tau)*target_var) for target_var, online_var in zip(variables_in_scope('target'), variables_in_scope('online'))]
    update_target_op = tf.group(*update_target)

    # Critic loss (MSE)
    q_loss = tf.reduce_mean((qa_onl - y_ph)**2)
    # Actor loss
    p_loss = -tf.reduce_mean(qd_onl)

    # Optimize the critic
    q_opt = tf.compat.v1.train.AdamOptimizer(cr_lr).minimize(q_loss)
    # Optimize the actor
    p_opt = tf.compat.v1.train.AdamOptimizer(ac_lr).minimize(p_loss, var_list=variables_in_scope('online/p_mlp'))

    def agent_op(o):
        a = np.squeeze(sess.run(p_onl, feed_dict={obs_ph:[o]}))
        return np.clip(a, [-160, -160, -160, -160], [160, 160, 160, 160])

    def agent_noisy_op(o, scale):
        action = agent_op(o)
        noisy_action = action + np.random.normal(loc=0.0, scale=scale, size=action.shape)
        return np.clip(noisy_action, [-160, -160, -160, -160], [160, 160, 160, 160])

    # Create a session and initialize the variables
    sess = tf.compat.v1.Session()
    sess.run(tf.compat.v1.global_variables_initializer())
    sess.run(init_target_op)

    #Create a saver object which will save all the variables
    saver = tf.compat.v1.train.Saver()

    # Some useful variables..
    step_count = 0
    last_q_update_loss = []
    last_p_update_loss = []
    batch_rew = []
    actual_epoch = 0
    obs = env.resety()

    # Initialize the buffer
    buffer = ExperiencedBuffer(buffer_size)

    for ep in range(num_epochs):
        g_rew = 0
        done = False
        actual_epoch += 1
        #env.clear_trajectory()

        while not done:
            step_count += 1
            print("             -=============- Epoch: ", actual_epoch,\
                    " Step: ", step_count, " -=============-")
            # If not gathered enough experience yet, act randomly
            if len(buffer) < min_buffer_size:
                act = env.sample()
                #env.clear_tratectory()
            else:
                act = agent_noisy_op(obs, 0.1)
            print(act)

            # Take a step in the environment
            obs2, rew, done = env.step(act, actual_epoch, step_count, False)
            print("Reward: ", rew)

            # Add the transition in the buffer
            buffer.add(obs.copy(), rew, act, obs2.copy(), done)

            obs = obs2
            g_rew += rew

            if len(buffer) > min_buffer_size:
                # sample a mini batch from the buffer
                mb_obs, mb_rew, mb_act, mb_obs2, mb_done = buffer.sample_minibatch(batch_size)

                # Compute the target values
                q_target_mb = sess.run(qd_tar, feed_dict={obs_ph:mb_obs2})
                y_r = np.array(mb_rew) + discount*(1-np.array(mb_done))*q_target_mb

                # optimize the critic
                _, q_train_loss = sess.run([q_opt, q_loss], feed_dict={obs_ph:mb_obs, y_ph:y_r, act_ph: mb_act})

                # optimize the actor
                _, p_train_loss = sess.run([p_opt, p_loss], feed_dict={obs_ph:mb_obs})

                # Soft update of the target networks
                sess.run(update_target_op)

            if done:
                obs = env.resety()
                batch_rew.append(g_rew)
                g_rew = 0

    saver.save(sess, args.train)

if __name__ == '__main__':
    if args.train:
        env = environment.arm()
        env.start()
        # env, hidden_sizes=[32], ac_lr=1e-2, cr_lr=1e-2, num_epochs=5000,\
        # buffer_size=200000, discount=0.99, batch_size=128, min_buffer_size=10000, tau=0.005):
        ddpg = threading.Thread(name = 'DDPG', target = DDPG, args = (env, [128, 128],\
                                3e-4, 4e-4, 5000, 200000, 0.99, 256, 5000, 0.005))
        ddpg.setDaemon(True)
        ddpg.start()
        environment.showPlot(env)
