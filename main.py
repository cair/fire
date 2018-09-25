from keras import Sequential
from keras.layers import Dense
import numpy as np
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
    model = Sequential()
    model.add(Dense(256, activation='relu', input_shape=(env2.action_space_shape,)))
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(env2.state_space.size, activation='linear'))
    model.compile(optimizer='adam', loss='mse')
    model.summary()
    pre_knowledge_q = np.array(tabular_q.Q.reshape((1, env2.state_space.size)))

    model.fit(np.expand_dims(env2.reset(), 0), pre_knowledge_q, epochs=200, batch_size=1, verbose=False, callbacks=[PlotLosses("Pretraining DQN")])
    model.save_weights('output/dqn_pretrained.h5')


    ############################################
    #
    # Training of Algorithms...
    #
    ###########################################
    #runner = Runner(env2, tabular_q2)
    #runner.run_episodes(100000)

    #algorithm_runner = AlgorithmRunner()
    #algorithm_runner.register(DQN)
    #algorithm_runner.register(VPG)

    #runner = Runner(env2, algorithm_runner)
    #runner.run_episodes(1000)


# Create clean environment


