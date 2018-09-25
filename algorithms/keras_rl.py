from rl.agents import SARSAAgent, DQNAgent
from rl.memory import SequentialMemory

class KerasRL:

    def __init__(self, model, output_shape):
        self.algorithms = []
        self.model = model
        self.output_shape = output_shape

    def add(self, algorithm):
        self.algorithms.append(algorithm)

    def dueling_dqn(self):
        return DQNAgent, dict(
            model=self.model, # Will be set later
            nb_actions=196,
            memory=SequentialMemory(limit=50000, window_length=1),
            nb_steps_warmup=1000,
            target_model_update=1e-4,
            enable_dueling_network=True,
            dueling_type='avg',
            policy=None,
            batch_size=128
        ), "QMP-DQN (Proposed)", "keras-rl"

    def sarsa(self):
        return SARSAAgent, dict(
            model=self.model, # Will be set later
            nb_actions=25,
            nb_steps_warmup=1000,
            train_interval=1,
            policy=None,
            name="SARSA"
        ), "SARSA", "keras-rl"

    def ddqn(self):
        return DQNAgent, dict(
            model=self.model,
            nb_actions=25,
            memory=SequentialMemory(limit=50000, window_length=1),
            nb_steps_warmup=1000,
            target_model_update=1e-4,
            policy=None,
            batch_size=32
        ), "DDQN", "keras-rl"

    def dqn(self):
        return DQNAgent, dict(
            model=self.model, # Will be set later
            nb_actions=self.output_shape,
            memory=SequentialMemory(limit=50000, window_length=1),
            nb_steps_warmup=1000,
            target_model_update=1e-3,
            enable_double_dqn=False,
            policy=None,
            batch_size=32
        ), "DQN", "keras-rl"