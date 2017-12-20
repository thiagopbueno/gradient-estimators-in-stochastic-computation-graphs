import tensorflow as tf

class MDP_RNNCell(tf.nn.rnn_cell.RNNCell):

    def __init__(self, mdp, policy):
        self.mdp = mdp
        self.policy = policy

    @property
    def action_size(self):
        return self.mdp.action_size
        
    @property
    def state_size(self):
        return self.mdp.state_size

    @property
    def output_size(self):
        return self.mdp.state_size + self.mdp.action_size + 2

    def __call__(self, inputs, state, scope=None):
        
        with self.mdp.graph.as_default():

            # timestep
            timestep = inputs

            # augment state by adding timestep to state vector
            state_t = tf.concat([state, timestep], axis=1)

            # add policy network with augmented state as input
            action = self.policy(state_t)

            # add MDP components to the RNN cell output
            next_state, log_prob = self.mdp.transition(state, action)
            reward = self.mdp.reward(next_state, action)

            # concatenate outputs
            outputs = tf.concat([reward, next_state, action, log_prob], axis=1)
            
        return outputs, next_state

    
class MDP_RNN(object):
    
    def __init__(self, mdp, policy):
        self.cell = MDP_RNNCell(mdp, policy)
        self.graph = mdp.graph
    
    def unroll(self, initial_state, timesteps):

        inputs = timesteps

        max_time = int(inputs.shape[1])
        state_size = self.cell.state_size
        action_size = self.cell.action_size

        with self.graph.as_default():
            
            # timesteps
            inputs = tf.placeholder_with_default(tf.constant(timesteps, name='timesteps'),
                                                 shape=(None, max_time, 1),
                                                 name='inputs')
            # initial cell state
            initial_state = tf.placeholder_with_default(tf.constant(initial_state),
                                                        shape=(None, self.cell.state_size),
                                                        name='initial_state')

            # dynamic time unrolling
            outputs, final_state = tf.nn.dynamic_rnn(
                self.cell,
                inputs,
                initial_state=initial_state,
                dtype=tf.float32)

            # gather reward, state and action series
            outputs = tf.unstack(outputs, axis=2)
            reward_series = tf.reshape(outputs[0], [-1, max_time, 1])
            state_series  = tf.stack(outputs[1:1+state_size], axis=2)
            action_series = tf.stack(outputs[1+state_size:1+state_size+action_size],  axis=2)
            log_prob_series = tf.stack(outputs[1+state_size+action_size:],  axis=2)
        
        return reward_series, state_series, action_series, log_prob_series, final_state
