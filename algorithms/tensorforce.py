from tensorforce.agents import PPOAgent


class TensorforceRL:


    def __init__(self, model, input_shape, output_shape):
        self.algorithms = []
        self.model = model
        self.input_shape = input_shape
        self.output_shape = output_shape

    def add(self, algorithm):
        self.algorithms.append(algorithm)

    def ppo(self):
        # Create a Proximal Policy Optimization agent
        return PPOAgent, dict(
            states=dict(type='float', shape=env2.render().shape),
            actions=dict(type='int', num_actions=self.output_shape),
            network=[
                dict(type='dense', size=256),
                dict(type='dense', size=1024),
                dict(type='dense', size=1024),
            ],
            batching_capacity=1000,
            step_optimizer=dict(
                type='adam',
                learning_rate=1e-4
            )
        ), "PPO-Agent", "tensorforce"
