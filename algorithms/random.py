import random

class RandomAgent:

    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape

    def act(self, a):
        action = random.randint(self.output_shape)
        return action

    def observe(self, s):
        pass