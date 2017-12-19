import time

import tensorflow as tf

class PolicyOptimizer(object):

    def minimize(self, epoch, path, show_progress=True):
        
        # saver
        with self.graph.as_default():
            saver = tf.train.Saver()
            
        with tf.Session(graph=self.graph) as sess:

            sess.run(tf.global_variables_initializer())

            start = time.time()

            losses = []
            for epoch_idx in range(epoch):

                # backprop and update weights
                _, loss, total = sess.run([self.train_step, self.loss, self.total])

                # store results
                losses.append(loss)

                # show information
                if show_progress:
                    print('Epoch {0:5}: loss = {1}\r'.format(epoch_idx, loss, total), end='')
            print()

            end = time.time()
            uptime = end - start
            print("Done in {0:.6f} sec.\n".format(uptime))

            # save model
            save_path = saver.save(sess, path)
            print("Model saved in file: %s" % save_path)
        
        return losses, saver, uptime
    
class SGDPolicyOptimizer(PolicyOptimizer):
    
    def __init__(self, graph, learning_rate, loss, total, gradients=None):
        self.graph = graph

        # performance metrics
        self.loss = loss
        self.total = total

        # hyperparameters
        self.learning_rate = learning_rate

        # optimization ops
        with self.graph.as_default():
            if gradients is None:
                self.train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
            else:
                updates = []
                for (gradient, variable) in gradients:
                    updates.append(variable.assign(variable - learning_rate * gradient))
                self.train_step = tf.group(*updates)
