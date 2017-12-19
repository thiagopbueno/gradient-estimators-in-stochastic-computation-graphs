import tensorflow as tf

def MSE_loss_function(graph, rewards):
    
    with graph.as_default():
        total = tf.reduce_sum(rewards, axis=1)
        loss  = tf.reduce_mean(tf.square(total))
    
    return total, loss
