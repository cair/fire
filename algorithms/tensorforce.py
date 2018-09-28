from tensorforce.agents import PPOAgent, VPGAgent, RandomAgent
import numpy as np

class TensorforceRL:


    def __init__(self, model, input_shape, output_shape):
        self.algorithms = []
        self.model = model
        self.tforce_model = self.keras_to_tensorforce(self.model)
        self.input_shape = input_shape
        self.output_shape = output_shape

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

    def ppo(self):
        # Create a Proximal Policy Optimization agent
        return PPOAgent, dict(
            states=dict(type='float', shape=self.input_shape),
            actions=dict(type='int', num_actions=self.output_shape),
            network=self.tforce_model,
            batching_capacity=128,
            step_optimizer=dict(
                type='adam',
                learning_rate=1e-6
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


