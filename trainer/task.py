# based on https://medium.com/google-cloud/how-to-do-time-series-prediction-using-rnns-and-tensorflow-and-cloud-ml-engine-2ad2eeb189e8

import tensorflow as tf
from tensorflow.contrib.learn import ModeKeys
from tensorflow.contrib.data.python.ops import grouping
from itertools import takewhile
import numpy as np
import string
import copy
from os import path

from trainer.data_loader import read_dataset
from trainer.language_model import CharacterLanguageModel, WordLanguageModel
from trainer.hooks import MaintainStateHook, SaveSummaryHook, PredictHook, RunAfterCheckpointHook, CheckpointSaverHookAfterFirst
from trainer.hyperparameters import Hyperparameters
from trainer.flags import define_flags
from trainer.network_model import create_network
from trainer.babble import babble

import logging
logging.getLogger().setLevel(logging.INFO)

logger = logging.getLogger('task')

FLAGS = define_flags()

hyperparameters = Hyperparameters(FLAGS)

def get_data_set(file_suffix, mode, hyperparameters):
  file_name = path.join(FLAGS.data_dir, FLAGS.data_prefix + file_suffix)

  return read_dataset(file_name, hyperparameters, mode)
  
def get_train_set(hyperparameters):
  return get_data_set('.train.txt', ModeKeys.TRAIN, hyperparameters)

def get_valid_set(hyperparameters):
  return get_data_set('.valid.txt', ModeKeys.TRAIN, hyperparameters)


def experiment_fn(hyperparameters):
  job_dir = getattr(FLAGS, 'job-dir')

  run_config = tf.estimator.RunConfig(
    save_summary_steps=100,
    log_step_count_steps=1000,
    save_checkpoints_secs=30 * 60
  )

  def babble_after_checkpoint():
    with tf.Session() as sess:
      logger.info('Generated: ' + babble(FLAGS, hyperparameters, 400))

  train_spec = tf.estimator.TrainSpec(
    input_fn=get_train_set(hyperparameters),
    max_steps=FLAGS.max_steps,
    hooks=[
      MaintainStateHook(), 
      SaveSummaryHook(FLAGS.summary_dir, hyperparameters), 
      CheckpointSaverHookAfterFirst(
        job_dir,
        save_secs=run_config.save_checkpoints_secs,
        save_steps=run_config.save_checkpoints_steps),
      RunAfterCheckpointHook(
        run_config, 
        babble_after_checkpoint
      )])

  eval_spec = tf.estimator.EvalSpec(
    input_fn=get_valid_set(hyperparameters), 
    # how often to run evaluation
    throttle_secs=2 * 60 * 60,
    hooks=[MaintainStateHook()],
    exporters=[])

  estimator = tf.estimator.Estimator(
    model_fn=create_network, 
    model_dir=job_dir,
    config=run_config,
    params=hyperparameters)

  tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

  logger.info('Generated: ' + babble(FLAGS, hyperparameters))

experiment_fn(hyperparameters)