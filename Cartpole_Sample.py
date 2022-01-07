# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 00:10:05 2019

@author: tanma
"""

import gym

env = gym.make('CartPole-v0')

env.reset()

for _ in range(1000):
    env.render()
    env.step(env.action_space.sample())

env.close()