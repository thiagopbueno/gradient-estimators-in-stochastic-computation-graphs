import tensorflow as tf

class PolicySimulator():

    def __init__(self, graph, path):
        self.graph = graph
        self.path = path

    def run(self, ops, initial_state, timesteps):
        with self.graph.as_default():
            saver = tf.train.Saver()

        with tf.Session(graph=self.graph) as sess:
            # restore learned policy model
            saver.restore(sess, self.path)

            # simulate MDP trajectories
            result = sess.run(ops, feed_dict={'inputs:0': timesteps, 'initial_state:0': initial_state})

        return result
