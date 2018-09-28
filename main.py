
from keras import Input, Model
from keras.layers import Dense, Flatten
import numpy as np


from algorithms.keras_rl import KerasRL
from algorithms.tensorforce import TensorforceRL
from environment import Environment
import scenario
from algorithms.tabular_q import TabularQ
from plot import PlotLosses
from runner import Runner

if __name__ == "__main__":
    EPISODES = 1000
    PRETRAIN_EPOCHS=200
    MAX_STEPS = 1000



# Construct Scenario
    scen = scenario.EasyScenario()

    ############################################
    #
    # Train Tabular Q-Matrix
    #
    ###########################################
    env = Environment(scenario=scen, type="shortest-path", debug=False)
    tabular_q = TabularQ(env.state_space_shape, env.action_space_shape)

    runner = Runner(env, tabular_q)
    runner.run_episodes(10000)

    ############################################
    #
    # Pretraining (Overfitting) of DQN Weights
    #
    ###########################################
    env2 = Environment(scenario=scen, type="shortest-evac", max_steps=MAX_STEPS, debug=False)

    the_input = Input((1,) + env2.render().shape)
    x = Flatten()(the_input)
    x = Dense(128, activation='relu', use_bias=False)(x)
    x = Dense(256, activation='relu', use_bias=False)(x)
    x = Dense(256, activation='relu', use_bias=False)(x)
    x = Dense(256, activation='relu', use_bias=False)(x)
    x = Dense(env2.state_space.size, activation='linear')(x)

    model = Model(inputs=[the_input], outputs=[x])
    model.compile(optimizer='adam', loss='mse')
    model.summary()

    pre_knowledge_q = np.array(tabular_q.Q.reshape((1, env2.state_space.size)))
    X = np.reshape(env2.reset(), (1, 1) + env2.state.shape)

    model.fit(X, pre_knowledge_q, epochs=PRETRAIN_EPOCHS, batch_size=1, verbose=False, callbacks=[PlotLosses("Pretraining DQN")])
    model.save_weights('output/dqn_pretrained.h5')

    ############################################
    #
    # Training of Algorithms...
    #
    ###########################################
    tensorforce = TensorforceRL(model=model, input_shape=env2.render().shape, output_shape=env2.state_space.size)
    tensorforce.add(tensorforce.ppo())
    tensorforce.add(tensorforce.vpg())
    tensorforce.add(tensorforce.random())

    keras_rl = KerasRL(model=model, input_shape=env2.render().shape, output_shape=env2.state_space.size)
    keras_rl.add(keras_rl.dqn())
    keras_rl.add(keras_rl.ddqn())
    keras_rl.add(keras_rl.dueling_dqn())
    keras_rl.add(keras_rl.sarsa())

    # Merge Keras-RL and Tensorforce Agents
    agents = keras_rl.algorithms + tensorforce.algorithms

    runner.run_algorithms(env2, model, agents, episodes=EPISODES, max_steps=MAX_STEPS)
