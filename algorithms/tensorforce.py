

class Tensorforce:


    def __init__(self):
        pass


    def ppo(self):
        return dict(
            states=dict(type='float', shape=input_shape),
            actions=dict(type='int', num_actions=25),
            network=[dict(type='dense', size=128, activation='relu'),
                     dict(type='dense', size=256, activation='relu'),
                     dict(type='dense', size=256, activation='relu'),
                     dict(type='dense', size=256, activation='relu'),
                     ],
            step_optimizer=dict(
                type='adam',
                learning_rate=1e-6
            ),
            batching_capacity=32,
            subsampling_fraction=0.25,
            likelihood_ratio_clipping=0.1,
            optimization_steps=25,
            update_mode=dict(
                unit='timesteps',
                batch_size=1,
                frequency=1
            ),
            name="PPO",
            type="tensorforce"
        ),