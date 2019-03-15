"""
This part of code is the DQN brain, which is a brain of the agent.
All decisions are made in here.
Using Tensorflow to build the neural network.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
Tensorflow: 1.0
gym: 0.7.3
"""

import numpy as np
from keras import initializers
from keras.layers import Input,Dense, Dropout
from keras.models import Model, load_model
from keras import optimizers

import time
import os

np.random.seed(1)

# Deep Q Network off-policy
class DeepQNetwork:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.01,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=300,
            memory_size=500,
            batch_size=32,
            e_greedy_increment=None,
            output_graph=False,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        # total learning step
        self.learn_step_counter = 0
        #self.sleep_flag = False

        # initialize zero memory [s, a, r, s_]
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))
        self.replace_count = 0

        self._build_net()

        if os.path.isfile("q_eval.h5"):
            self.q_eval.load_weights("q_eval.h5")
            self.q_next.load_weights("q_eval.h5")
            print("load_weights finished")

        self.q_eval.summary()


    def _create_model(self):
        input_x = Input(shape = (self.n_features,))
        x = Dense(10, kernel_initializer=initializers.random_normal(stddev=0.3),
            bias_initializer = initializers.Constant(0.1),activation='relu')(input_x)
        predictions = Dense(self.n_actions, kernel_initializer=initializers.random_normal(stddev=0.3),
            bias_initializer = initializers.Constant(0.1))(x)
        model = Model(inputs=input_x, outputs=predictions)
        model.compile(optimizer=optimizers.RMSprop(lr=self.lr), loss='mean_squared_error')
        return model

    def _build_net(self):
        self.q_eval = self._create_model()
        self.q_next = self._create_model()


    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0

        transition = np.hstack((s, [a, r], s_))

        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition

        self.memory_counter += 1

    def choose_action(self, observation):
        # to have batch dimension when feed into tf placeholder
        observation = observation[np.newaxis, :]

        if np.random.uniform() < self.epsilon:
            # forward feed the observation and get q value for every actions
            actions_value = self.q_eval.predict(observation)
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def learn(self):
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            weights = self.q_eval.get_weights()
            self.q_next.set_weights(weights)
            #self.sleep_flag = True
            self.replace_count += 1
        print('replace_count:' + str(self.replace_count))

        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)

        #print(sample_index)
        batch_memory = self.memory[sample_index, :]

        #print("batch_memory:")
        #print(batch_memory)

        #print("batch_memory[:, -self.n_features:]:")
        #print(batch_memory[:, -self.n_features:])

        #print("batch_memory[:, :self.n_features]:")
        #print(batch_memory[:, :self.n_features])

        q_next = self.q_next.predict(batch_memory[:, -self.n_features:])

        # change q_target w.r.t q_eval's action
        q_target = self.q_eval.predict(batch_memory[:, :self.n_features])


        #print("q_target_before:")
        #print(np.hstack([batch_memory[:, :2], q_target]))

        #print("type(q_target)")
        #print(type(q_target))
        #print(q_target)

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features + 1]

        #print("[batch_index, eval_act_index]:")
        #print([batch_index, eval_act_index])

        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)

        #print("q_target fix:")
        #print(np.hstack([batch_memory[:, :2], q_target]))

        '''
        if(self.sleep_flag):
            time.sleep(1)
        '''

        """
        For example in this batch I have 2 samples and 3 actions:
        q_eval =
        [[1, 2, 3],
         [4, 5, 6]]

        q_target = q_eval =
        [[1, 2, 3],
         [4, 5, 6]]

        Then change q_target with the real q_target value w.r.t the q_eval's action.
        For example in:
            sample 0, I took action 0, and the max q_target value is -1;
            sample 1, I took action 2, and the max q_target value is -2:
        q_target =
        [[-1, 2, 3],
         [4, 5, -2]]

        So the (q_target - q_eval) becomes:
        [[(-1)-(1), 0, 0],
         [0, 0, (-2)-(6)]]

        We then backpropagate this error w.r.t the corresponding action to network,
        leave other action as error=0 cause we didn't choose it.
        """

        # train eval network
        self.q_eval.fit(batch_memory[:, :self.n_features], q_target)

        # increasing epsilon
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

    def predict_action(self, observation):
        observation = observation[np.newaxis, :]

        actions_value = self.q_next.predict(observation)
        print(observation)
        print(actions_value)

        action = np.argmax(actions_value)
        return action

    def save_model(self):
        self.q_eval.summary()
        self.q_eval.save_weights("q_eval.h5")
