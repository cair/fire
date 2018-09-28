from tensorforce.agents import PPOAgent, VPGAgent, RandomAgent, DQNAgent
import numpy as np

class TensorforceRL:


    def __init__(self, model, input_shape, output_shape, max_steps):
        self.algorithms = []
        self.model = model
        self.tforce_model = self.keras_to_tensorforce(self.model)
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.max_steps = max_steps

    def keras_to_tensorforce(self, model):
        network_spec = []
        # Convert Keras model to tensorforce dict
        first_dense = True
        prev_weights = None
        dense_layers = len([0 for _ in model.layers if _.name.split("_")[0] == "dense"])
        for i, keras_layer in enumerate(model.layers):

            layer_type = keras_layer.name.split("_")[0]
            if layer_type == "dense" and i <= dense_layers:
                if first_dense:
                    first_dense = False
                    prev_weights = keras_layer.get_weights()
                else:
                    network_spec.append(dict(type="dense", size=keras_layer.input_shape[-1], bias=False ,  weights=np.vstack(prev_weights)))
                    prev_weights = keras_layer.get_weights()

        return network_spec

    def add(self, algorithm):
        self.algorithms.append(algorithm)

    def dqn(self):
        # Create a Proximal Policy Optimization agent
        return DQNAgent, dict(
            states=dict(type='float', shape=self.input_shape),
            actions=dict(type='int', num_actions=self.output_shape),
            network=self.tforce_model,
            batching_capacity=16,
            update_mode=dict(
                unit="timesteps",
                batch_size=64,
                frequency=4
            ),
            memory= dict(
                type="replay",
                include_next_states=True,
                capacity=1000*16),
            optimizer=dict(
                type='adam',
                learning_rate=1e-3
            ),
            actions_exploration=dict(
                type="epsilon_anneal",
                initial_epsilon=0.9,
                final_epsilon=0.0,
                timesteps=self.max_steps * 100
            ),
            double_q_model=True,
            target_sync_frequency=500,
            target_update_weight=0.1
        ), "QMP-DQN", "tensorforce"

    def ppo(self):
        # Create a Proximal Policy Optimization agent
        return PPOAgent, dict(
            states=dict(type='float', shape=self.input_shape),
            actions=dict(type='int', num_actions=self.output_shape),
            network=self.tforce_model,
            batching_capacity=16,
            update_mode=dict(
                unit="timesteps",
                batch_size=16,
                frequency=400
            ),
            #memory=dict(
            #    type="latest",
            #    include_next_states=True,
            #    capacity=5000
            #),
            step_optimizer=dict(
                type='adam',
                learning_rate=1e-3
            ),
            discount=0.95,
            entropy_regularization=0.01,
            likelihood_ratio_clipping=1.0,
            baseline_mode="states",
            baseline=dict(
                type="mlp",
                sizes=[16, 16]
            ),
            actions_exploration=dict(
                type="epsilon_anneal",
                initial_epsilon=0.9,
                final_epsilon=0.0,
                timesteps=100000
            ),
            baseline_optimizer=dict(
                type="multi_step",
                optimizer=dict(
                    type="adam",
                    learning_rate=1e-3
                ),
                num_steps=4
            )

        ), "PPO", "tensorforce"


    def vpg(self):
        return VPGAgent, dict(
            states=dict(type='float', shape=self.input_shape),
            actions=dict(type='int', num_actions=self.output_shape),
            network=self.tforce_model,
            optimizer=dict(
                type='adam',
                learning_rate=1e-6
            ),
            batching_capacity=128,
            update_mode=dict(
                unit='timesteps',
                batch_size=1,
                frequency=1
            ),
        ), "VPG", "tensorforce"


    def random(self):
        return RandomAgent, dict(
            states=dict(type='float', shape=self.input_shape),
            actions=dict(type='int', num_actions=self.output_shape)
        ), "Random", "tensorforce"


