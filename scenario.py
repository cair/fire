import numpy as np

class BaseScenario:

    def __init__(self):
        self.state_space = None
        self.action_space = None
        self.fire_locations = None
        self.terminal_states = None
        self.state_space_shape = None
        self.action_space_shape = None


    def apply(self, env):
        env.state_space = np.copy(self.state_space)
        env.action_space = np.copy(self.action_space)
        env.state_space_shape = self.state_space_shape
        env.action_space_shape = self.action_space_shape
        env.fire_locations = np.copy(self.fire_locations)
        env.terminal_states = np.copy(self.terminal_states)



class EasyScenario(BaseScenario):

    def __init__(self):
        super().__init__()
        self.state_space = np.array([
            [0, 1, 1, 0, 0, 0, 0, 0],
            [1, 0, 0, 1, 1, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 0, 0, 1],
            [0, 1, 0, 0, 0, 1, 0, 1],
            [1, 0, 0, 0, 1, 0, 1, 1],
            [0, 0, 1, 0, 0, 1, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 1],
        ])
        self.action_space = len(self.state_space)
        self.action_space_shape = self.action_space
        self.state_space_shape = (len(self.state_space), )



        self.fire_locations = [2, 3, 4]
        self.terminal_states = [7]

class HardScenario(BaseScenario):

    def __init__(self):
        super().__init__()
        self.state_space = np.array([
            [1, 1, 1, 1, 0],
            [1, 1, 1, 1, 0],
            [1, 1, 1, 0, 1],
            [1, 1, 0, 1, 1],
            [0, 0, 1, 1, 0]
        ])
        self.action_space = len(self.state_space)

        self.fire_locations = [2]
        self.terminal_states = [11, 12, 13]
