import time
import numpy as np
import keras.callbacks
from keras.optimizers import Adam
from tqdm import tqdm

from plot import PlotPerformance


class Callback(keras.callbacks.Callback):

    def __init__(self, agent_name, agent_type, perf_plot, plot_every):
        self.agent_name = agent_name
        self.episode = 0
        self.perf_plot = perf_plot
        self.perf_plot.new(self.agent_name, agent_type)
        self.plot_every = plot_every

    def _set_env(self, env):

        self.env = env

    def on_episode_begin(self, episode, logs={}):
        pass

    def on_episode_end(self, episode, logs={}):
        self.episode += 1
        self.perf_plot.log(self.episode, logs["nb_episode_steps"])

        if self.episode % self.plot_every == 1:
            print("Plot!")
            self.perf_plot.to_file()

    def on_train_end(self, logs=None):
        self.perf_plot.to_file()

    def on_step_begin(self, step, logs={}):
       pass

    def on_step_end(self, step, logs={}):
        pass

    def on_action_begin(self, action, logs={}):
        pass

    def on_action_end(self, action, logs={}):
        pass


class Runner:

    def __init__(self, env, algorithm):
        self.env = env
        self.algorithm = algorithm

    def run_episode(self, max_steps):
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

            if steps > max_steps:
                break

        return steps

    def run_episodes(self, episodes, max_steps):
        for i in tqdm(range(episodes)):
            self.run_episode(max_steps)

    @staticmethod
    def run_algorithms(environment, model, agents, max_steps=1000, episodes=1000, plot_every=100):
        perf_plot = PlotPerformance()
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
                history = agent.fit(environment, nb_steps=episodes*250, nb_max_episode_steps=max_steps, visualize=False, verbose=2, callbacks=[Callback(agent_name, agent_type, perf_plot, plot_every)])

            elif agent_type == "tensorforce":

                agent = agent_class(**agent_spec)
                action_space = [0 for x in range(environment.state_space.size)]
                # Get prediction from agent, execute
                perf_plot.new(agent_name, agent_type)
                for episode in range(episodes):
                    t = False
                    environment.reset()
                    steps = np.zeros(episodes)
                    while t is False:
                        action = agent.act(environment.render())
                        action_space[action] += 1
                        steps[episode] += 1

                        if steps[episode] >= max_steps:
                            t = True
                            break

                        s1, r, t, _ = environment.step(action)
                        agent.observe(reward=r, terminal=t)

                    perf_plot.log(episode, steps[episode])

                    if episode % plot_every == 1:
                        perf_plot.to_file()

                    print("Steps=%s, Episode=%s, Action-Dist=%s" % (steps[episode], episode, action_space))
                perf_plot.to_file()
