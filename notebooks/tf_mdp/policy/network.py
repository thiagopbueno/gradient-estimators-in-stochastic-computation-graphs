import tensorflow as tf
import functools

class PolicyNetwork(object):
    
    def __init__(self, graph, layers, limits=1.0):
        self.graph = graph
        self.policy = functools.partial(self.__build_network, layers, limits)
    
    def __call__(self, state):
        return self.policy(state)
    
    def __build_network(self, layers, limits, state):

        with self.graph.as_default():

            with tf.variable_scope('policy'):

                # hidden layers
                outputs = state
                for i, n_h in enumerate(layers[1:]):
                    if i != len(layers)-2:
                        activation = tf.nn.relu
                    else:
                        activation = tf.nn.tanh

                    outputs = tf.layers.dense(outputs,
                                              units=n_h,
                                              activation=activation,
                                              kernel_initializer=tf.glorot_normal_initializer(),
                                              name="layer"+str(i+1))

                # add action limits over last tanh layer
                action = tf.constant(limits) * outputs

        return action
