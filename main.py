from environment import Environment
from algorithms.tabular_q import TabularQ
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

    # Setup environment
    env = Environment()
    tabular_q = TabularQ((14,), 14)

    runner = Runner(env, tabular_q)
    runner.run_episodes(10000)

    print(tabular_q.Q)

    # Create clean environment


