import gym
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers.merge import Add, Multiply
from keras.optimizers import Adam
import keras.backend as K

import tensorflow as tf

import random
from collections import deque


class ActorCritic:
    def __init__(self, env, sess):
        self.env = env
        self.sess = sess

        self.learning_rate = 0.001
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.gamma = 0.95
        self.tau = 0.125
        self.memory = deque(maxlen=2000)

        # Fix Tensorflow import bug in Keras to allow dropout to work
        K.set_learning_phase(1)

        # Actor initialization
        self.actor_state_input, self.actor_model = self.create_actor_model()
        _, self.target_actor_model = self.create_actor_model()

        self.actor_critic_grad = tf.placeholder(tf.float32, [None, self.env.action_space.n])

        actor_model_weights = self.actor_model.trainable_weights
        self.actor_grads = tf.gradients(self.actor_model.output, actor_model_weights, -self.actor_critic_grad)
        grads = zip(self.actor_grads, actor_model_weights)
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(grads)

        # Critic initialization
        self.critic_state_input, self.critic_action_input, self.critic_model = self.create_critic_model()
        _, _, self.target_critic_model = self.create_critic_model()

        self.critic_grads = tf.gradients(self.critic_model.output, self.critic_action_input)

        #init for later gradient calculations
        self.sess.run(tf.initialize_all_variables())

    def create_actor_model(self):
        state_input = Input(shape=self.env.observation_space.shape)
        # Using a Conv Network to reduce dimensions down
        h1 = Conv2D(filters=16, kernel_size=(2,2), strides=1, padding="same", activation='relu', use_bias=True)(state_input)
        h1 = MaxPooling2D(pool_size=(2,2), strides=None)(h1)
        h1 = Conv2D(filters=32, kernel_size=(2,2), strides=1, activation='relu', use_bias=True)(h1)
        h1 = MaxPooling2D(pool_size=(2, 2), strides=None)(h1)
        h1 = Conv2D(filters=64, kernel_size=(2, 2), strides=1, activation='relu', use_bias=True)(h1)
        h1 = MaxPooling2D(pool_size=(2, 2), strides=None)(h1)
        h1 = GlobalAveragePooling2D()(h1)
        h1 = Dropout(0.2)(h1)
        output = Dense(self.env.action_space.n,
                       activation='relu')(h1)

        model = Model(input=state_input, output=output)
        print(model.summary())

        adam = Adam(lr=self.learning_rate)
        model.compile(loss="mse", optimizer=adam)
        return state_input, model

    def create_critic_model(self):
        state_input = Input(shape=self.env.observation_space.shape)
        state_h1 = Conv2D(filters=16, kernel_size=(2,2), strides=1, padding="same", activation='relu', use_bias=True)(state_input)
        state_h1 = MaxPooling2D(pool_size=(2, 2), strides=None)(state_h1)
        state_h1 = Conv2D(filters=32, kernel_size=(2, 2), strides=1, activation='relu', use_bias=True)(state_h1)
        state_h1 = MaxPooling2D(pool_size=(2, 2), strides=None)(state_h1)
        state_h1 = Conv2D(filters=64, kernel_size=(2, 2), strides=1, activation='relu', use_bias=True)(state_h1)
        state_h1 = MaxPooling2D(pool_size=(2, 2), strides=None)(state_h1)
        state_h1 = GlobalAveragePooling2D()(state_h1)
        state_h2 = Dense(48)(state_h1)

        #action_input = Input(shape=self.env.action_space.shape)
        action_input = Input(shape=(self.env.action_space.n,))
        action_h1 = Dense(48)(action_input)

        merged = Add()([state_h2, action_h1])
        merged_h1 = Dense(24, activation='relu')(merged)
        output = Dense(1, activation='relu')(merged_h1)
        model = Model(input=[state_input, action_input], output=output)
        print(model.summary())

        adam = Adam(lr=self.learning_rate)
        model.compile(loss="mse", optimizer=adam)
        return state_input, action_input, model

    def remember(self, cur_state, action, reward, new_state, done):
        self.memory.append([cur_state, action, reward, new_state, done])

    def train(self):
        batch_size = 32
        if len(self.memory) < batch_size:
            return

        rewards = []
        samples = random.sample(self.memory, batch_size)
        self._train_critic(samples)
        self._train_actor(samples)

    def _train_critic(self, samples):
        for sample in samples:
            cur_state, action, reward, new_state, done = sample
            if not done:
                target_action = self.target_actor_model.predict(new_state)
                future_reward = self.target_critic_model.predict([new_state, target_action])[0][0]
                reward += self.gamma * future_reward
            self.critic_model.fit([cur_state, action], np.array([reward]), verbose=0)

    def _train_actor(self, samples):
        for sample in samples:
            cur_state, action, reward, new_state, _ = sample
            predicted_action = self.actor_model.predict(cur_state)
            grads = self.sess.run(self.critic_grads, feed_dict={
                self.critic_state_input: cur_state,
                self.critic_action_input: predicted_action
            })[0]

            self.sess.run(self.optimize, feed_dict={
                self.actor_state_input: cur_state,
                self.actor_critic_grad: grads
            })

    def _update_actor_target(self):
        actor_model_weights = self.actor_model.get_weights()
        actor_target_weights = self.target_critic_model.get_weights()

        for i in range(len(actor_target_weights)):
            actor_target_weights[i] = actor_model_weights[i]
        self.target_critic_model.set_weights(actor_target_weights)

    def _update_critic_target(self):
        critic_model_weights = self.critic_model.get_weights()
        critic_target_weights = self.target_critic_model.get_weights()

        for i in range(len(critic_target_weights)):
            critic_target_weights[i] = critic_model_weights[i]
        self.target_critic_model.set_weights(critic_target_weights)

    def update_target(self):
        self._update_actor_target()
        self._update_critic_target()

    def act(self, cur_state):
        self.epsilon *= self.epsilon_decay
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        return np.argmax(self.actor_model.predict(cur_state))


def main():
    sess = tf.Session()
    K.set_session(sess)
    env = gym.make("SpaceInvaders-v0")
    actor_critic = ActorCritic(env, sess)

    num_trials = 10000
    trial_len = 500

    cur_state = env.reset()
    for trial in range(num_trials):
        env.reset()
        print("trial {}".format(trial))
        for _ in range(trial_len):
            if trial % 100 == 0:
                env.render()
            cur_state = cur_state.reshape((1, env.observation_space.shape[0], env.observation_space.shape[1],
                                           env.observation_space.shape[2]))
            action = actor_critic.act(cur_state)

            new_state, reward, done, _ = env.step(action)
            new_state = new_state.reshape((1, env.observation_space.shape[0], env.observation_space.shape[1],
                                           env.observation_space.shape[2]))

            # one-hot encode action for NN
            action = np.array([action])
            action = np.eye(env.action_space.n)[action]

            actor_critic.remember(cur_state, action, reward, new_state, done)

            actor_critic.train()

            if done:
                break

if __name__ == "__main__":
    main()
