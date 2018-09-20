
import numpy as np
import random

class TabularQ:


    def __init__(self, state_shape, action_shape, lr=0.99, discount=0.8, e_start=1.0, e_stop=0.1, e_steps=1000000, e_decay_enabled=True):
        if type(state_shape) is not tuple:
            raise TypeError("state_shape must be a tuple: (14, )")
        if type(action_shape) is not int:
            raise TypeError("action_shape must be a integer: 14")


        self.Q = np.zeros(shape=state_shape + (action_shape, ))
        self.action_shape = action_shape
        self.state_shape = state_shape
        self.lr = lr
        self.discount = discount
        self.e_start = e_start
        self.e_stop = e_stop
        self.e_steps = e_steps
        self.e_decay = (e_start - e_stop) / e_steps if e_decay_enabled else 0
        self.e = self.e_start

    def train(self, s, a, s1, r):
        # Get old val
        # Q(state, action) = R(state, action) + Gamma * Max[Q(next state, all actions)]
        #env.Q[s,a] = env.Q[s, a] + lr * (r + (discount *  np.max(env.Q[s1])) - env.Q[s, a])
        self.Q[s,a] = self.lr * (r + (self.discount *  np.max(self.Q[s1])))
        #env.Q[s1, a] = r + discount * np.max(env.Q[s1])


    def action(self, s):
        if random.uniform(0, 1) < self.e:
            a = random.randint(0, self.action_shape-1) # 2
        else:
            a = np.argmax(self.Q[s])
        self.e -= self.e_decay

        return a

