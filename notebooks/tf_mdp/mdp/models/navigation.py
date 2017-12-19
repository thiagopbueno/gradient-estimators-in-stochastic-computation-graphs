import tf_mdp.mdp.models.mdp as mdp

import numpy as np
import tensorflow as tf

class Navigation(mdp.MDP):

    def __init__(self, graph, grid, alpha_min=0, alpha_max=10):
        self.graph = graph

        self.ndim = grid["ndim"]

        with self.graph.as_default():

            # grid constants
            self.__size = tf.constant(grid["size"], dtype=tf.float32, name="grid_size")
            self.__goal = tf.constant(grid["goal"], dtype=tf.float32, name="goal")

            # numerical constants
            self.__0_00 = tf.constant(0.0, dtype=tf.float32)
            self.scale_min = tf.constant(2*np.pi/360 * alpha_min, dtype=tf.float32, name="scale_min")
            self.scale_max = tf.constant(2*np.pi/360 * alpha_max, dtype=tf.float32, name="scale_max")

    @property
    def action_size(self):
        return self.ndim
    
    @property
    def state_size(self):
        return self.ndim
        
    def transition(self, state, action, noise=None):

        with self.graph.as_default():

            # noise
            velocity = tf.norm(action, axis=1, keep_dims=True, name="velocity")
            max_velocity = tf.sqrt(2.0, name="max_velocity")

            scale = tf.maximum(self.scale_min, self.scale_max / max_velocity * velocity, name="scale")
            loc = tf.constant(0.0, name="loc")
            noise = tf.distributions.Normal(loc=loc, scale=scale, name="noise")

            # alpha = tf.stop_gradient(noise.sample(name="alpha"))
            alpha = noise.sample(name="alpha")
            log_prob = noise.log_prob(alpha, name="log_prob")
            self.graph.add_to_collection('stochastic', alpha)
            
            # apply rotation noise to generate next state
            cos, sin = tf.cos(alpha), tf.sin(alpha)
            rotation_matrix = tf.stack([cos, -sin, sin, cos], axis=1)
            rotation_matrix = tf.reshape(rotation_matrix, [-1, 2, 2], name="rotation_matrix")
            noisy_action = tf.matmul(rotation_matrix, tf.reshape(action, [-1, 2, 1]))
            noisy_action = tf.reshape(noisy_action, [-1, 2], name="noisy_action")
            
            # next position
            p = tf.add(state, noisy_action, name="p")

            # avoid getting out of map
            next_state = tf.clip_by_value(p, self.__0_00, self.__size, name="next_state")

        return next_state, log_prob

    def reward(self, state, action):

        with self.graph.as_default():

            # norm L-2 (euclidean distance)
            r = -tf.sqrt(tf.reduce_sum(tf.square(state - self.__goal), axis=1, keep_dims=True))

        return r
