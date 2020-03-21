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