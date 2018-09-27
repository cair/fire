import numpy as np
import random



class EnvironmentLogic:

    @staticmethod
    def shortest_path(env, s, a):
        r = -1
        t = False

        if env.state_space[s, a] == 1:
            env.state = a
        else:
            r = -1

        if env.state in env.terminal_states:
            r = 1
            t = True

        return env.state, r, t, {}

    @staticmethod
    def shortest_evac(env, s, a):

        # Unpack action
        from_room = int(a % env.action_space_shape)    # % is the "modulo operator", the remainder of i / width;
        to_room = int(a / env.action_space_shape)    # where "/" is an integer division

        if to_room in env.fire_locations:
            r = -10
        elif env.state[from_room] <= 0 or env.state[to_room] >= 10:
            r = -1
        elif from_room == to_room:
            r = -0.2
        elif to_room in env.terminal_states and env.state_space[from_room, to_room] == 1 and env.state[from_room] > 0:
            # 1. to_room is a exit, 2. its a legal action, 3. and there is a person in the from_room
            r = 1
            env.state[from_room] -= 1
        elif env.state_space[from_room,to_room] == 1:     #legal moves
            env.state[from_room] -=1
            env.state[to_room] +=1
            r = -0.01
        elif env.state_space[from_room, to_room] == 0:     #illegal moves
            r = -1

        # Reset number of people at the terminal states
        for i in env.terminal_states:
            env.state[i] = 0

        t = True if np.sum(env.state) == 0 else False

        if env.max_steps is not None and env.steps > env.max_steps:
            t = True

        return env.state, r, t, {}

class Environment:
    def __init__(self, scenario=None, max_steps=None, type='shortest-path', debug=True):
        self._logic = None
        self.type = type
        self.max_steps = max_steps
        self.steps = 0
        self.debug = debug

        self.state_space = None
        self.state_space_shape = None
        self.action_space = None
        self.action_space_shape = None
        self.fire_locations = None
        self.terminal_states = None
        self.state = None

        self.path = []
        self.path_stats = {}

        scenario.apply(self) # Copy over the scenario template
        self.scenario = scenario
        self.reset()

    def reset(self):
        # Clear constructed path
        if self.debug:
            self.summary()

            # Counter for paths found
            proposed_path = "=>".join(self.path)
            if proposed_path not in self.path_stats:
                self.path_stats[proposed_path] = 1
            else:
                self.path_stats[proposed_path] += 1

        self.path.clear()
        self.steps = 0

        if self.type == "shortest-path":
            # Spawn at random pos every episode

            self.scenario.shortest_path_state(self)

            # Add initial state to path
            self.path.append(str(self.state))

            # Set environment logic
            self._logic = EnvironmentLogic.shortest_path

        elif self.type == "shortest-evac":

            self.scenario.shortest_evac_state(self)

            self._logic = EnvironmentLogic.shortest_evac

        else:
            raise TypeError("scenario is invalid, must be shortest-path or shortest-evac")

        self.steps = 0
        return self.render()

    def summary(self):
        print("Path=%s, Steps=%s" % (self.path, self.steps))

    def step(self, a):

        output = self._logic(self, self.state, a)
        self.steps += 1
        self.path.append(str(self.state))

        return output


    def render(self):
        return self.state

