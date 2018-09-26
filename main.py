import time

from keras import Sequential, Input, Model
from keras.layers import Dense, Flatten
import numpy as np
from keras.optimizers import Adam

from algorithms.keras_rl import KerasRL
from environment import Environment
import scenario
from algorithms.tabular_q import TabularQ
from plot import PlotPerformance, PlotLosses
from util import pretty_dict


class Runner:

    def __init__(self, env, algorithm):
        self.env = env
        self.algorithm = algorithm

    def run_episode(self):
        terminal = False
        steps = 0
        s = self.env.reset()
        while not terminal:

            # Observe the state
            s = self.env.render()



            # Predict best action
            a = self.algorithm.action(s)


            # Perform action
            s1, r, t, _ = self.env.step(a)



            # Train
            self.algorithm.train(s, a, s1, r)

            terminal = t
            steps += 1
            s = s1

        return steps

    def run_episodes(self, n):
        for i in range(n):
            self.run_episode()


if __name__ == "__main__":

    # Construct Scenario
    scen = scenario.EasyScenario()

    ############################################
    #
    # Train Tabular Q-Matrix
    #
    ###########################################
    env = Environment(scenario=scen, type="shortest-path", debug=False)
    tabular_q = TabularQ(env.state_space_shape, env.action_space_shape)

    #runner = Runner(env, tabular_q)
    #runner.run_episodes(1000)


    ############################################
    #
    # Pretraining (Overfitting) of DQN Weights
    #
    ###########################################
    env2 = Environment(scenario=scen, type="shortest-evac", debug=True)

    the_input = Input((1, ) + env2.render().shape)
    #flatten = Flatten()(the_input)
    x = Dense(256, activation='relu')(the_input)
    x = Dense(1024, activation='relu')(x)
    x = Dense(1024, activation='relu')(x)
    x = Dense(1024, activation='relu')(x)
    x = Dense(env2.state_space.size, activation='linear')(x)
    x = Flatten()(x)

    model = Model(inputs=[the_input], outputs=[x])
    model.compile(optimizer='adam', loss='mse')
    model.summary()

    pre_knowledge_q = np.array(tabular_q.Q.reshape((1, env2.state_space.size)))
    X = np.reshape(env2.reset(), (1, 1) + env2.state.shape)

    for layer in model.weights:
        layer_type = layer.name.split("_")[0]
        layer_size = layer.shape[0]

        print(layer_size, layer_type, dir(layer), layer)

    model.fit(X, pre_knowledge_q, epochs=200, batch_size=1, verbose=False, callbacks=[PlotLosses("Pretraining DQN")])
    model.save_weights('output/dqn_pretrained.h5')


    ############################################
    #
    # Training of Algorithms...
    #
    ###########################################
    perf_plot = PlotPerformance()
    EPISODES = 50
    keras_rl = KerasRL(model=model, output_shape=env2.state_space.size)
    keras_rl.add(keras_rl.dqn())


    # Merge Keras-RL and Tensorforce Agents
    agents = keras_rl.algorithms


    for agent_class, agent_spec, agent_name, agent_type in agents:
        print(agent_class, agent_spec, agent_type)

        if agent_type == "keras-rl":
            model.load_weights('output/dqn_pretrained.h5')
            model.compile(optimizer='adam', loss='mse')

            # Agent Init
            agent = agent_class(**agent_spec)
            print("Starting experiment for %s." % agent_name)

            # Agent Train
            agent.compile(Adam(lr=1e-2), metrics=['mse'])
            history = agent.fit(env2, nb_steps=EPISODES*250, nb_max_episode_steps=1000, visualize=False, verbose=2)

            # Fetch Train Summary
            summary_step = history.history["nb_episode_steps"][:EPISODES]

            # Plotting
            perf_plot.new(agent_name)
            start_time = time.time()
            for episode, steps in enumerate(summary_step):
                perf_plot.log(episode, steps)
            print("Model: %s, Average Steps: %s, Minimum Steps: %s, Time: %.3f secs" % (name, np.average(summary_step), np.amin(summary_step), (time.time() - start_time)))





    #runner = Runner(env2, tabular_q2)
    #runner.run_episodes(100000)

    #algorithm_runner = AlgorithmRunner()
    #algorithm_runner.register(DQN)
    #algorithm_runner.register(VPG)

    #runner = Runner(env2, algorithm_runner)
    #runner.run_episodes(1000)


# Create clean environment


