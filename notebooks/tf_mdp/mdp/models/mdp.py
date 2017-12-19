import abc

class MDP(metaclass=abc.ABCMeta):
    
    @abc.abstractproperty
    def action_size(self):
        return
    
    @abc.abstractproperty
    def state_size(self):
        return

    @abc.abstractmethod
    def transition(self, state, action, noise=None):
        return

    @abc.abstractmethod
    def reward(self, state, action):
        return
