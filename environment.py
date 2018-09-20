import numpy as np
import random

class RewardModels:

    @staticmethod
    def shortest_path(env, s, a):
        r = -1
        t = False

        if env.state_space[s, a] == 1:
            env.s = a
        else:
            r = -100

        if env.s in env.terminal_states:
            r = 100
            t = True




        """
            self.n_steps += 1
        # Unpack action
        from_room = int(action % 14)    # % is the "modulo operator", the remainder of i / width;
        to_room = int(action / 14)    # where "/" is an integer division
        t = False

        if to_room == self.F: #fire location
            r = -1000
        elif to_room == 11 and self.adjacency[from_room,to_room] == 1 and self.state[from_room] > 0:      #exit
            r = 100
            self.state[from_room] -=1
        elif to_room == 12 and self.adjacency[from_room,to_room] == 1 and self.state[from_room] > 0:      #exit
            r = 100
            self.state[from_room] -=1
        elif to_room == 13 and self.adjacency[from_room,to_room] == 1 and self.state[from_room] > 0:      #exit
            r = 100
            self.state[from_room] -=1
        elif (self.state[from_room] <= 0 or self.state[to_room] >= 10) and from_room != to_room:    #bottleneck
            r = -100
        elif self.adjacency[from_room,to_room] == 1:     #legal moves
            self.state[from_room] -=1
            self.state[to_room] +=1
            r = -1
        elif self.adjacency[from_room,to_room] == 0:     #illegal moves
            r = -100

        self.state[11] = 0                           #reset number of people at the exit
        self.state[12] = 0                           #reset number of people at the exit
        self.state[13] = 0                           #reset number of people at the exit
        t = True if np.sum(self.state) == 0 else False
        #print(self.state)

        if self.max_steps is not None and self.n_steps > self.max_steps:
            t = True

        return self.render(), r, t, _

        """

        return env.s, r, t, {}

class Environment:
    def __init__(self, max_steps=None, scenario='shortest-path'):
        self.scenario = scenario
        self.max_steps = max_steps
        self.steps = 0
        self.action_space = 14
        self.state_space = np.array([
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
        ])

        self.fire_locations = [2]
        self.terminal_states = [11, 12, 13]

        self.state = None

        self.reset()

    def reset(self):
        if self.scenario == "shortest-path":
            # Spawn at random pos every episode
            self.state = random.choice([i for i in range(0, self.action_space) if i not in self.terminal_states + self.fire_locations])
        elif self.scenario == "shortest-evac":
            self.state = np.array([10,10,10,10,10,10,10,10,10,10,10,0,0,0])
        else:
            raise TypeError("scenario is invalid, must be shortest-path or shortest-evac")

        self.steps = 0
        return self.render()


    def step(self, a):
        pass


    def render(self):
        return self.state

