import argparse
import gym
import csv
import time
import os
import datetime
import math
import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp

from gym import wrappers, logger
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.layers import Lambda, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras import backend as K
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras.utils import plot_model


def softplusk(x):
    """Some implementations use a modified softplus 
        to ensure that the stddev is never zero
    Argument:
        x (tensor): activation input
    """
    return K.softplus(x) + 1e-10


class PolicyAgent:
    def __init__(self, env):
        """Implements the models and training of 
            Policy Gradient Methods
        Argument:
            env (Object): OpenAI gym environment
        """

        self.env = env
        # entropy loss weight
        self.beta = 0.0
        # value loss for all policy gradients except A2C
        self.loss = self.value_loss
        
        # s,a,r,s' are stored in memory
        self.memory = []

        # for computation of input size
        self.state = env.reset()
        self.state_dim = env.observation_space.shape[0]
        self.state = np.reshape(self.state, [1, self.state_dim])
        self.build_autoencoder()


    def reset_memory(self):
        """Clear the memory before the start 
            of every episode
        """
        self.memory = []


    def remember(self, item):
        """Remember every s,a,r,s' in every step of the episode
        """
        self.memory.append(item)


    def action(self, args):
        """Given mean and stddev, sample an action, clip 
            and return
            We assume Gaussian distribution of probability 
            of selecting an action given a state
        Argument:
            args (list) : mean, stddev list
        Return:
            action (tensor): policy action
        """
        mean, stddev = args
        dist = tfp.distributions.Normal(loc=mean, scale=stddev)
        action = dist.sample(1)
        action = K.clip(action,
                        self.env.action_space.low[0],
                        self.env.action_space.high[0])
        return action


    def logp(self, args):
        """Given mean, stddev, and action compute
            the log probability of the Gaussian distribution
        Argument:
            args (list) : mean, stddev action, list
        Return:
            logp (tensor): log of action
        """
        mean, stddev, action = args
        dist = tfp.distributions.Normal(loc=mean, scale=stddev)
        logp = dist.log_prob(action)
        return logp


    def entropy(self, args):
        """Given the mean and stddev compute 
            the Gaussian dist entropy
        Argument:
            args (list) : mean, stddev list
        Return:
            entropy (tensor): action entropy
        """
        mean, stddev = args
        dist = tfp.distributions.Normal(loc=mean, scale=stddev)
        entropy = dist.entropy()
        return entropy


    def build_autoencoder(self):
        """Autoencoder to convert states into features
        """
        # first build the encoder model
        inputs = Input(shape=(self.state_dim, ), name='state')
        feature_size = 32
        x = Dense(256, activation='relu')(inputs)
        x = Dense(128, activation='relu')(x)
        feature = Dense(feature_size, name='feature_vector')(x)

        # instantiate encoder model
        self.encoder = Model(inputs, feature, name='encoder')
        self.encoder.summary()
        plot_model(self.encoder,
                   to_file='encoder.png', 
                   show_shapes=True)

        # build the decoder model
        feature_inputs = Input(shape=(feature_size,), 
                               name='decoder_input')
        x = Dense(128, activation='relu')(feature_inputs)
        x = Dense(256, activation='relu')(x)
        outputs = Dense(self.state_dim, activation='linear')(x)

        # instantiate decoder model
        self.decoder = Model(feature_inputs, 
                             outputs, 
                             name='decoder')
        self.decoder.summary()
        plot_model(self.decoder, 
                   to_file='decoder.png', 
                   show_shapes=True)

        # autoencoder = encoder + decoder
        # instantiate autoencoder model
        self.autoencoder = Model(inputs, 
                                 self.decoder(self.encoder(inputs)),
                                 name='autoencoder')
        self.autoencoder.summary()
        plot_model(self.autoencoder, 
                   to_file='autoencoder.png', 
                   show_shapes=True)

        # Mean Square Error (MSE) loss function, Adam optimizer
        self.autoencoder.compile(loss='mse', optimizer='adam')


    def train_autoencoder(self, x_train, x_test):
        """Training the autoencoder using randomly sampled
            states from the environment
        Arguments:
            x_train (tensor): autoencoder train dataset
            x_test (tensor): autoencoder test dataset
        """
        # train the autoencoder
        batch_size = 32
        self.autoencoder.fit(x_train,
                             x_train,
                             validation_data=(x_test, x_test),
                             epochs=10,
                             batch_size=batch_size)


    def build_actor_critic(self):
        """4 models are built but 3 models share the
            same parameters. hence training one, trains the rest.
            The 3 models that share the same parameters 
                are action, logp, and entropy models. 
            Entropy model is used by A2C only.
            Each model has the same MLP structure:
            Input(2)-Encoder-Output(1).
            The output activation depends on the nature 
                of the output.
        """
        inputs = Input(shape=(self.state_dim, ), name='state')
        self.encoder.trainable = False
        x = self.encoder(inputs)
        mean = Dense(1,
                     activation='linear',
                     kernel_initializer='zero',
                     name='mean')(x)
        stddev = Dense(1,
                       kernel_initializer='zero',
                       name='stddev')(x)
        # use of softplusk avoids stddev = 0
        stddev = Activation('softplusk', name='softplus')(stddev)
        action = Lambda(self.action,
                        output_shape=(1,),
                        name='action')([mean, stddev])
        self.actor_model = Model(inputs, action, name='action')
        self.actor_model.summary()
        plot_model(self.actor_model, 
                   to_file='actor_model.png', 
                   show_shapes=True)

        logp = Lambda(self.logp,
                      output_shape=(1,),
                      name='logp')([mean, stddev, action])
        self.logp_model = Model(inputs, logp, name='logp')
        self.logp_model.summary()
        plot_model(self.logp_model, 
                   to_file='logp_model.png', 
                   show_shapes=True)

        entropy = Lambda(self.entropy,
                         output_shape=(1,),
                         name='entropy')([mean, stddev])
        self.entropy_model = Model(inputs, entropy, name='entropy')
        self.entropy_model.summary()
        plot_model(self.entropy_model, 
                   to_file='entropy_model.png', 
                   show_shapes=True)

        value = Dense(1,
                      activation='linear',
                      kernel_initializer='zero',
                      name='value')(x)
        self.value_model = Model(inputs, value, name='value')
        self.value_model.summary()
        plot_model(self.value_model, 
                   to_file='value_model.png', 
                   show_shapes=True)


        # logp loss of policy network
        loss = self.logp_loss(self.get_entropy(self.state), 
                              beta=self.beta)
        optimizer = RMSprop(lr=1e-3)
        self.logp_model.compile(loss=loss, optimizer=optimizer)

        optimizer = Adam(lr=1e-3)
        self.value_model.compile(loss=self.loss, optimizer=optimizer)


    def logp_loss(self, entropy, beta=0.0):
        """logp loss, the 3rd and 4th variables 
            (entropy and beta) are needed by A2C 
            so we have a different loss function structure
        Arguments:
            entropy (tensor): Entropy loss
            beta (float): Entropy loss weight
        Return:
            loss (tensor): computed loss
        """
        def loss(y_true, y_pred):
            return -K.mean((y_pred * y_true) \
                    + (beta * entropy), axis=-1)

        return loss


    def value_loss(self, y_true, y_pred):
        """Typical loss function structure that accepts 
            2 arguments only
           This will be used by value loss of all methods 
            except A2C
        Arguments:
            y_true (tensor): value ground truth
            y_pred (tensor): value prediction
        Return:
            loss (tensor): computed loss
        """
        loss = -K.mean(y_pred * y_true, axis=-1)
        return loss


    def save_weights(self, 
                     actor_weights, 
                     encoder_weights, 
                     value_weights=None):
        """Save the actor, critic and encoder weights
            useful for restoring the trained models
        Arguments:
            actor_weights (tensor): actor net parameters
            encoder_weights (tensor): encoder weights
            value_weights (tensor): value net parameters
        """
        self.actor_model.save_weights(actor_weights)
        self.encoder.save_weights(encoder_weights)
        if value_weights is not None:
            self.value_model.save_weights(value_weights)


    def load_weights(self,
                     actor_weights, 
                     value_weights=None):
        """Load the trained weights
           useful if we are interested in using 
                the network right away
        Arguments:
            actor_weights (string): filename containing actor net
                weights
            value_weights (string): filename containing value net
                weights
        """
        self.actor_model.load_weights(actor_weights)
        if value_weights is not None:
            self.value_model.load_weights(value_weights)

    
    def load_encoder_weights(self, encoder_weights):
        """Load encoder trained weights
           useful if we are interested in using 
            the network right away
        Arguments:
            encoder_weights (string): filename containing encoder net
                weights
        """
        self.encoder.load_weights(encoder_weights)

    
    def act(self, state):
        """Call the policy network to sample an action
        Argument:
            state (tensor): environment state
        Return:
            act (tensor): policy action
        """
        action = self.actor_model.predict(state)
        return action[0]


    def value(self, state):
        """Call the value network to predict the value of state
        Argument:
            state (tensor): environment state
        Return:
            value (tensor): state value
        """
        value = self.value_model.predict(state)
        return value[0]


    def get_entropy(self, state):
        """Return the entropy of the policy distribution
        Argument:
            state (tensor): environment state
        Return:
            entropy (tensor): entropy of policy
        """
        entropy = self.entropy_model.predict(state)
        return entropy[0]


class REINFORCEAgent(PolicyAgent):
    def __init__(self, env):
        """Implements the models and training of 
           REINFORCE policy gradient method
        Arguments:
            env (Object): OpenAI gym environment
        """
        super().__init__(env) 

    def train_by_episode(self):
        """Train by episode
           Prepare the dataset before the step by step training
        """
        # only REINFORCE and REINFORCE with baseline
        # use the ff code
        # convert the rewards to returns
        rewards = []
        gamma = 0.99
        for item in self.memory:
            [_, _, _, reward, _] = item
            rewards.append(reward)
        # rewards = np.array(self.memory)[:,3].tolist()

        # compute return per step
        # return is the sum of rewards from t til end of episode
        # return replaces reward in the list
        for i in range(len(rewards)):
            reward = rewards[i:]
            horizon = len(reward)
            discount =  [math.pow(gamma, t) for t in range(horizon)]
            return_ = np.dot(reward, discount)
            self.memory[i][3] = return_

        # train every step
        for item in self.memory:
            self.train(item, gamma=gamma)


    def train(self, item, gamma=1.0):
        """Main routine for training 
        Arguments:
            item (list) : one experience unit
            gamma (float) : discount factor [0,1]
        """
        [step, state, next_state, reward, done] = item

        # must save state for entropy computation
        self.state = state

        discount_factor = gamma**step
        delta = reward

        # apply the discount factor as shown in Algortihms
        # 10.2.1, 10.3.1 and 10.4.1
        discounted_delta = delta * discount_factor
        discounted_delta = np.reshape(discounted_delta, [-1, 1])
        verbose = 1 if done else 0

        # train the logp model (implies training of actor model
        # as well) since they share exactly the same set of
        # parameters
        self.logp_model.fit(np.array(state),
                            discounted_delta,
                            batch_size=1,
                            epochs=1,
                            verbose=verbose)
        
        
class REINFORCEBaselineAgent(REINFORCEAgent):
    def __init__(self, env):
        """Implements the models and training of 
           REINFORCE w/ baseline policy 
           gradient method
        Arguments:
            env (Object): OpenAI gym environment
        """
        super().__init__(env) 


    def train(self, item, gamma=1.0):
        """Main routine for training 
        Arguments:
            item (list) : one experience unit
            gamma (float) : discount factor [0,1]
        """
        [step, state, next_state, reward, done] = item

        # must save state for entropy computation
        self.state = state

        discount_factor = gamma**step

        # reinforce-baseline: delta = return - value
        delta = reward - self.value(state)[0] 

        # apply the discount factor as shown in Algorithms
        # 10.2.1, 10.3.1 and 10.4.1
        discounted_delta = delta * discount_factor
        discounted_delta = np.reshape(discounted_delta, [-1, 1])
        verbose = 1 if done else 0

        # train the logp model (implies training of actor model
        # as well) since they share exactly the same set of
        # parameters
        self.logp_model.fit(np.array(state),
                            discounted_delta,
                            batch_size=1,
                            epochs=1,
                            verbose=verbose)

        # train the value network (critic)
        self.value_model.fit(np.array(state),
                             discounted_delta,
                             batch_size=1,
                             epochs=1,
                             verbose=verbose)
        

class A2CAgent(PolicyAgent):
    def __init__(self, env):
        """Implements the models and training of 
           A2C policy gradient method
        Arguments:
            env (Object): OpenAI gym environment
        """
        super().__init__(env) 
        # beta of entropy used in A2C
        self.beta = 0.9
        # loss function of A2C value_model is mse
        self.loss = 'mse'


    def train_by_episode(self, last_value=0):
        """Train by episode 
           Prepare the dataset before the step by step training
        Arguments:
            last_value (float): previous prediction of value net
        """
        # implements A2C training from the last state
        # to the first state
        # discount factor
        gamma = 0.95
        r = last_value
        # the memory is visited in reverse as shown
        # in Algorithm 10.5.1
        for item in self.memory[::-1]:
            [step, state, next_state, reward, done] = item
            # compute the return
            r = reward + gamma*r
            item = [step, state, next_state, r, done]
            # train per step
            # a2c reward has been discounted
            self.train(item)