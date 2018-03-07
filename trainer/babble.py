# run this task to generate text based on a trained model.

import tensorflow as tf
from itertools import takewhile
import numpy as np
import copy
from os import path

from trainer.hooks import MaintainStateHook, PredictHook
from trainer.hyperparameters import Hyperparameters
from trainer.flags import define_flags
from trainer.network_model import create_network

def babble(FLAGS, hyperparameters, text_length=4000):  
  job_dir = getattr(FLAGS, 'job-dir')
  
  # during predictions, we have a batch size of 1 (because we predict 
  # one sentence at a time) and a depth of 1 (the point of unrolling 
  # the network is to ease training; it's not helpful during prediction)
  predict_hyperparams = copy.copy(hyperparameters)
  predict_hyperparams.batch_size = 1
  predict_hyperparams.unroll_steps = 1

  estimator = tf.estimator.Estimator(
    model_fn=create_network, 
    model_dir=job_dir,
    config=tf.estimator.RunConfig(),
    params=predict_hyperparams)

  predict_hook = PredictHook('is currently accepting applications for', hyperparameters)

  prediction_seq = estimator.predict(predict_hook.predict_input_fn,
    hooks=[MaintainStateHook(), predict_hook])

  # iterate through the prediction sequence for text_length number of characters

  for char in takewhile(lambda x: len(predict_hook.predicted_str) < text_length, prediction_seq):
    pass

  # this is the generated text.
  return predict_hook.predicted_str
