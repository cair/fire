import time
import numpy as np
from keras.optimizers import Adam

from plot import PlotPerformance


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

    @staticmethod
    def run_algorithms(environment, model, agents, max_steps=1000, episodes=1000):
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
                history = agent.fit(environment, nb_steps=episodes*250, nb_max_episode_steps=max_steps, visualize=False, verbose=2)

                # Fetch Train Summary
                summary_step = history.history["nb_episode_steps"][:episodes]

                # Plotting
                perf_plot.new(agent_name)
                start_time = time.time()
                for episode, steps in enumerate(summary_step):
                    perf_plot.log(episode, steps)
                print("Model: %s, Average Steps: %s, Minimum Steps: %s, Time: %.3f secs" % (agent_name, np.average(summary_step), np.amin(summary_step), (time.time() - start_time)))


            elif agent_type == "tensorforce":

                agent = agent_class(**agent_spec)

                # Get prediction from agent, execute
                perf_plot.new(agent_name)
                for episode in range(episodes):
                    t = False
                    environment.reset()
                    steps = np.zeros(episodes)
                    while t is False:
                        action = agent.act(environment.render())
                        steps[episode] += 1

                        if steps[episode] >= max_steps:
                            t = True
                            break

                        s1, r, t, _ = environment.step(action)
                        agent.observe(reward=r, terminal=t)

                    perf_plot.log(episode, steps[episode])
                    print("Steps=%s, Episode=%s" % (steps[episode], episode))

