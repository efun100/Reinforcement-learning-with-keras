"""
The double DQN based on this paper: https://arxiv.org/abs/1509.06461

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
Tensorflow: 1.0
gym: 0.8.0
"""

import numpy as np
from keras import initializers
from keras.layers import Input,Dense, Dropout
from keras.models import Model, load_model
from keras import optimizers

np.random.seed(1)

class DoubleDQN:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.02,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=200,
            memory_size=3000,
            batch_size=32,
            e_greedy_increment=None,
            double_q=True,
            sess=None,
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

        self.double_q = double_q    # decide to use double q or not

        self.learn_step_counter = 0
        self.memory = np.zeros((self.memory_size, n_features*2+2))
        self._build_net()

        #self.cost_his = []

    def _build_net(self):
        def build_layers(n_l1):
            input_x = Input(shape = (self.n_features,))
            x = Dense(n_l1, kernel_initializer=initializers.random_normal(stddev=0.3),
                bias_initializer = initializers.Constant(0.1),activation='relu')(input_x)
            predictions = Dense(self.n_actions, kernel_initializer=initializers.random_normal(stddev=0.3),
                bias_initializer = initializers.Constant(0.1))(x)
            model = Model(inputs=input_x, outputs=predictions)
            model.compile(optimizer=optimizers.RMSprop(lr=self.lr), loss='mean_squared_error')
            return model
        # ------------------ build net ------------------
        n_l1 = 20

        self.q_eval = build_layers(n_l1)
        self.q_next = build_layers(n_l1)

    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        transition = np.hstack((s, [a, r], s_))
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def choose_action(self, observation):
        observation = observation[np.newaxis, :]
        actions_value = self.q_eval.predict(observation)
        action = np.argmax(actions_value)

        if not hasattr(self, 'q'):  # record action value it gets
            self.q = []
            self.running_q = 0
        self.running_q = self.running_q*0.99 + 0.01 * np.max(actions_value)
        self.q.append(self.running_q)

        if np.random.uniform() > self.epsilon:  # choosing action
            action = np.random.randint(0, self.n_actions)
        return action

    def predict_action(self, observation):
        observation = observation[np.newaxis, :]
        actions_value = self.q_eval.predict(observation)
        action = np.argmax(actions_value)
        return action

    def learn(self):
        if self.learn_step_counter % self.replace_target_iter == 0:
            weights = self.q_eval.get_weights()
            self.q_next.set_weights(weights)
            print('\ntarget_params_replaced\n')

        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        q_next = self.q_next.predict(batch_memory[:, -self.n_features:])
        q_eval4next = self.q_eval.predict(batch_memory[:, -self.n_features:])

        q_target = self.q_eval.predict(batch_memory[:, :self.n_features])

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features + 1]

        if self.double_q:
            max_act4next = np.argmax(q_eval4next, axis=1)        # the action that brings the highest value is evaluated by q_eval
            selected_q_next = q_next[batch_index, max_act4next]  # Double DQN, select q_next depending on above actions
        else:
            selected_q_next = np.max(q_next, axis=1)    # the natural DQN

        q_target[batch_index, eval_act_index] = reward + self.gamma * selected_q_next

        self.q_eval.fit(batch_memory[:, :self.n_features], q_target)

        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

    def save_model(self, name):
        self.q_eval.summary()
        self.q_eval.save(name)

    def load_model(self, name):
        self.q_eval.load_weights(name)
