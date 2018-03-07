# run this task to generate text based on a trained model.

import tensorflow as tf
from itertools import takewhile
import numpy as np
import copy
from os import path

from trainer.babble import babble
from trainer.hyperparameters import Hyperparameters
from trainer.flags import define_flags

import logging
logging.getLogger().setLevel(logging.INFO)

FLAGS = define_flags()

hyperparameters = Hyperparameters(FLAGS)

print(babble(FLAGS, hyperparameters, 4000))