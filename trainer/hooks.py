import numpy as np
import tensorflow as tf
from trainer.network_model import *
from tensorflow.python.training import session_run_hook
from tensorflow.python.training.basic_session_run_hooks import SecondOrStepTimer, CheckpointSaverHook
from tensorflow.python.training import training_util

class MaintainStateHook(tf.train.SessionRunHook):
  """ We need to maintain the state of the LSTM during both training and prediction; otherwise
  it has no way of keeping track of "where it is" in a sequence. So this hook reads the LSTM
  state at the end of a step and feeds it back into the LSTM at the start of the next step.
  Note that this requires that each LSTM unit receives the slices of input in order, which is
  what the data loader makes sure.
  """
  last_state = None

  def before_run(self, run_context):
    # feed back the output state of the last step as initial state to this step
    feed_dict = {}

    if self.last_state is not None:
      # the LSTM has two states that need to be preserved, c and h
      # `get_collection` will return a list of variables, one for each layer of RNN
      # assign all those variables the corresponding value from `last_state`
      feed_dict = {
        **dict(zip(tf.get_collection(INITIAL_LSTM_STATE_C), self.last_state['c'])),
        **dict(zip(tf.get_collection(INITIAL_LSTM_STATE_H), self.last_state['h']))
      }

    # variables to be calculated during the session. again, 
    # `get_collection` will return a list of values, one for each RNN layer
    fetches = {
      'c': tf.get_collection(LSTM_STATE_C),
      'h': tf.get_collection(LSTM_STATE_H)
    }

    return tf.train.SessionRunArgs(
      fetches=fetches,
      feed_dict=feed_dict,
      options=None)

  def after_run(self, run_context, run_values):
    # `results` has the same structure as `fetches` above.
    self.last_state = run_values.results

class PredictHook(tf.train.SessionRunHook):
  """
  At prediction time (when we are generating a string based on the model), the
  model needs to be fed the last generated token as input 
  (in addition to the last state, of course). 
  """

  """ The entire prediction so far."""
  predicted_str = ''

  seed_str = ''

  """ The last predicted token. This is the input to the next step. """
  last_token = ''

  def __init__(self, seed_str, hyperparameters):
    # The first token we start with is "end of string", 
    # indicating that we are at the start of a new string.
    self.last_token = hyperparameters.language_model.EOS
    self.seed_token_ids = hyperparameters.language_model.tokenize_str(seed_str)
    self.language_model = hyperparameters.language_model

  def begin(self):
    # This is the tensor that contains the predicted  
    # class. It is drawn at random from the probability distribution
    # indicated by the output logits. We could also have fetched the 
    # logits themselves to get more than one prediction.
    self.prediction_class_id = tf.get_collection(PREDICTION_CLASS_ID)[0]

  def before_run(self, run_context):
    # Tell TF we want to calculate the prediction.
    return tf.train.SessionRunArgs(
      self.prediction_class_id)
      
  def after_run(self, run_context, run_values):
    lm = self.language_model

    predicted_token = lm.token(run_values.results[0][0])

    self.last_token = predicted_token
    self.predicted_str = lm.append(self.predicted_str, predicted_token)

  
  def predict_input_fn(self):
    """
    While during training we have an input source in the form of training data,
    during prediction the input source is in fact the previous prediction (the 
    start of the predicted string).

    This method is an input function to be used with the `Estimator` for 
    predictions feeding back the previously generated string.

    Note the difference to `MaintainStateHook`, that feeds the state using
    `SessionRunArgs`, because there is already an input source during training, 
    which is the training data.
    """
    def gen():
      # We can produce an infinite sequence of predictions
      while True:
        if self.seed_token_ids:
          last_seed_id = self.seed_token_ids[0]
          self.seed_token_ids = self.seed_token_ids[1:]

          # Input has rank 2 (batch_size x unroll_steps) but both dimensions are size 1.
          yield ([[last_seed_id]])
        else:
          # Input has rank 2 (batch_size x unroll_steps) but both dimensions are size 1.
          yield ([[self.language_model.id_of_token(self.last_token)]])
    
    # Input is 1x1-dimensional. Note that for this to work, we need to set
    # `batch_size` and `unroll_steps` to 1 during prediction. 
    # It would of course be possible to generalize it to multiple batches.
    return tf.data.Dataset.from_generator(
      gen, (tf.int32), (tf.TensorShape([1,1])))



class SaveSummaryHook(tf.train.SessionRunHook):
  """ Saves TensorBoard data at the end of the training run. """
  def __init__(self, output_dir, hyperparameters):
    self.output_dir = output_dir

  def begin(self):
    self.train_writer = tf.summary.FileWriter(self.output_dir, tf.get_default_graph())
    self.merged = tf.summary.merge_all()

  def before_run(self, run_context):
    return tf.train.SessionRunArgs(
      {
        'summary': self.merged,
        'step': tf.train.get_global_step()
      },
      feed_dict={},
      options=None)
  
  def after_run(self, run_context, run_values):
    self.train_writer.add_summary(run_values.results['summary'], run_values.results['step'])

class RunAfterCheckpointHook(session_run_hook.SessionRunHook):
  """ Runs a certain callback function right after a checkpoint has been saved. 
      We use this to generate some text at regular intervals during the training to show the progress. 
      Note that it restores the model from a checkpoint, which is why it needs to happen with the same 
      interval as checkpoint saving. """
  def __init__(self, run_config, callback):
    self._timer = SecondOrStepTimer(
                              every_secs=run_config.save_checkpoints_secs,
                              every_steps=run_config.save_checkpoints_steps)
    self.callback = callback
    self.is_first_run = True

  def begin(self):
    self._global_step_tensor = training_util._get_or_create_global_step_read()  # pylint: disable=protected-access

  def after_run(self, run_context, run_values):
    global_step = run_context.session.run(self._global_step_tensor)

    if self._timer.should_trigger_for_step(global_step):
      self._timer.update_last_triggered_step(global_step)

      # the timer will tell us that it needs to trigger on the very first run, which does not make sense.
      if not self.is_first_run:
        self.callback()
      else:
        self.is_first_run = False
 
class CheckpointSaverHookAfterFirst(CheckpointSaverHook):
  """ The default CheckointSaverHook stores a checkpoint the very first thing it does during training.
      That seems a bit meaningless. This one skips that save.
  """
  def _save(self, session, step):
    if getattr(self, 'do_save', False):
      super()._save(session, step)
    else:
      self.do_save = True