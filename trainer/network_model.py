import numpy as np
import tensorflow as tf
import tensorflow.contrib.rnn as rnn
from tensorflow.contrib.learn import ModeKeys
from trainer.debug import debug_print

# TODO: take from flags
TF_DATA_TYPE = tf.float32

# collection name
PREDICTION_LOGITS = 'prediction_logits'
PREDICTION_CLASS_ID = 'prediction_class_id'

LSTM_STATE_C = 'lstm_state_c'
LSTM_STATE_H = 'lstm_state_h'
INITIAL_LSTM_STATE_C = 'initial_lstm_state_c'
INITIAL_LSTM_STATE_H = 'initial_lstm_state_h'

def create_network(features, labels, mode, params):
  """ Create the inference model """
  # we can't change the parameter names in the function declaration or TF gets angry,
  # so let's rename them here
  hyperparameters = params

  # this is the current slice of tokens (chars/words)
  tokens = features
  
  # this is the next slice of tokens, one ahead of `tokens`. Note that `next_tokens`
  # and tokens will overlap; they are only off by one token but they are `unroll_steps`
  # long. None if we are doing prediction
  next_tokens = labels
  unroll_steps = hyperparameters.unroll_steps
  num_classes = hyperparameters.language_model.num_classes

  # ... as opposed to prediction. during prediction, labels are not present
  # and there's no need to calculate loss.
  is_training = mode == ModeKeys.TRAIN or mode == ModeKeys.EVAL

  dropout = hyperparameters.dropout if mode == ModeKeys.TRAIN else 0

  with tf.control_dependencies(
      [tf.assert_equal(tf.shape(tokens), [hyperparameters.batch_size, unroll_steps])] + 
      [tf.assert_equal(tf.shape(next_tokens), [hyperparameters.batch_size, unroll_steps])] if is_training else []):
    embedding_size = hyperparameters.embedding_size

    if embedding_size:
      embeddings = tf.get_variable('input_embeddings',
          [num_classes, embedding_size])
      inputs = tf.nn.embedding_lookup(embeddings, tokens)
    else:
      # convert from tokens (e.g. "2") to one-hot representation (e.g. "[0, 0, 1, 0]")
      inputs = tf.one_hot(tokens, num_classes)

    # note: the batch size is 1 during prediction
    batch_size = hyperparameters.batch_size

    def create_lstm_cell(layer):
      if hyperparameters.layer_norm:
        if hyperparameters.num_proj:
          raise Exception('No support for layer normalization together with projection layer.')

        cell = rnn.LayerNormBasicLSTMCell(
          hyperparameters.lstm_state_size, 
          # here, we use the local variable dropout that is set to 0
          # if we are evaluating.
          dropout_keep_prob=1-dropout,
          layer_norm=hyperparameters.layer_norm)
      else:
        if hyperparameters.num_proj:
          cell = rnn.LSTMCell(
            hyperparameters.lstm_state_size,
            num_proj=hyperparameters.num_proj
          )
        else:
          cell = rnn.LSTMBlockCell(
            hyperparameters.lstm_state_size, forget_bias=0)

        if dropout > 0:
          cell = rnn.DropoutWrapper(cell, output_keep_prob=1-dropout)

      return cell

    rnn_cell = rnn.MultiRNNCell(
        [create_lstm_cell(layer) for layer in range(hyperparameters.layers)])

    initial_state = rnn_cell.zero_state(batch_size, TF_DATA_TYPE)
    
    # TODO: switch to dynamic_rnn?

    # `static_rnn` requires `inputs` to be a list of one-dimensional 
    # (one-hot encoded) values
    inputs = tf.unstack(inputs, num=unroll_steps, axis=1)

    # outputs has shape [ unroll_steps, batch_size, lstm_state_size (or num_proj if set) ] 
    outputs, states = rnn.static_rnn(rnn_cell, inputs, initial_state=initial_state, dtype=TF_DATA_TYPE)

    output_size = hyperparameters.lstm_state_size if not hyperparameters.num_proj else hyperparameters.num_proj

    outputs = tf.reshape(outputs, [-1, output_size])
    # and now [ unroll_steps * batch_size, lstm_state_size ]

    # let output be a linear combination of the activation of the last layer of the RNN
    softmax_w = tf.get_variable(
        "softmax_w", [output_size, num_classes], dtype=TF_DATA_TYPE)
    softmax_b = tf.get_variable("softmax_b", [num_classes], dtype=TF_DATA_TYPE)

    prediction_logits = tf.nn.xw_plus_b(outputs, softmax_w, softmax_b)
    # prediction_logits is now [ unroll_steps * batch_size, CLASSES ]

    prediction_logits = tf.reshape(prediction_logits, [unroll_steps, -1, num_classes], name='logits')
    # prediction_logits is now [ unroll_steps, batch_size, CLASSES ] 

    prediction_logits = tf.transpose(prediction_logits, [ 1, 0, 2 ])
    # prediction_logits is now [ batch_size, unroll_steps, CLASSES ] 

    # sample the probability distribution represented by the logits to arrive at a 
    # predicted next character (only used during prediction, not training)
    # -1 means we take the output of the last step, i.e. the predicted last 
    # character. On the other hand, we set `unroll_steps` to 1 during prediction,
    # so we could replace it with 0.
    prediction_class_id = tf.multinomial(prediction_logits[:,-1,:], num_samples=1)

    # store the various state tensors in collections
    # where the hooks can retrieve them later
    tf.add_to_collection(PREDICTION_LOGITS, prediction_logits)
    tf.add_to_collection(PREDICTION_CLASS_ID, prediction_class_id)

    for single_layer_initial_state in initial_state:
      tf.add_to_collection(INITIAL_LSTM_STATE_C, single_layer_initial_state.c)
      tf.add_to_collection(INITIAL_LSTM_STATE_H, single_layer_initial_state.h)

    for single_layer_state in states:
      tf.add_to_collection(LSTM_STATE_C, single_layer_state.c)
      tf.add_to_collection(LSTM_STATE_H, single_layer_state.h)

    if is_training:
      # the loss is the difference between the predicted distribution
      # and the actual next tokens.
      loss = tf.contrib.seq2seq.sequence_loss(
        prediction_logits,
        next_tokens,
        tf.ones(tf.shape(next_tokens), 
        dtype=TF_DATA_TYPE),
        average_across_timesteps=True,
        average_across_batch=True)

      learning_rate = tf.train.exponential_decay(
        hyperparameters.learning_rate, 
        tf.train.get_global_step(),
        decay_steps=hyperparameters.lr_decay_steps, 
        decay_rate=hyperparameters.lr_decay_rate, 
        staircase=True)

      train_op = tf.contrib.layers.optimize_loss(
        loss=loss,
        global_step=tf.train.get_global_step(),
        clip_gradients=hyperparameters.clip_gradients,
        learning_rate=learning_rate,
        optimizer=hyperparameters.optimizer)
      
      perplexity = tf.exp(loss)
      
      tf.summary.scalar('perplexity', perplexity)

      eval_metric_ops = {
        'eval_avg_loss': tf.metrics.mean(loss),
        'perplexity': tf.metrics.mean(perplexity)
      }
    else:
      loss = None
      train_op = None
      eval_metric_ops = None

    return tf.estimator.EstimatorSpec(
      mode=mode,
      predictions=
        { 
          PREDICTION_LOGITS: prediction_logits,
          PREDICTION_CLASS_ID: prediction_class_id 
        },
      loss=loss,
      train_op=train_op,
      eval_metric_ops=eval_metric_ops,
      export_outputs=
        { "predicted": tf.estimator.export.PredictOutput(prediction_logits) })
