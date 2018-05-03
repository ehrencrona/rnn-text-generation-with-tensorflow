import tensorflow as tf

from os import path
import logging

logger = logging.getLogger('flags')

def define_flags():
  flags = tf.flags

  flags.DEFINE_bool('use_fp16', False,
                    'Train using 16-bit floats instead of 32bit floats')
  flags.DEFINE_string('data_prefix', 'text',
                      'File name prefix of the training and validation data (suffix is ".train.txt" and ".valid.txt").')
  # tensorflow expects "job-dir", not "job_dir"
  flags.DEFINE_string('job-dir', path.join('jobs', '%JOB%'),
                      'Where the finished model is stored.')
  flags.DEFINE_string('summary_dir', path.join('summaries', '%JOB%'),
                      'Where the summary data for TensorFlow is stored.')
  flags.DEFINE_string('data_dir', 'data',
                      'Path to the training and validation data.')
  flags.DEFINE_integer('lstm_state_size', 5,
                      'Number of sentences processed in parallel during training.')
  flags.DEFINE_integer('batch_size', 20, 
                      'Batches (sentences) trained in parallel.')
  flags.DEFINE_integer('embedding_size', None,
                      'The size of the initial embedding layer. Skips embeddings if not set.')
  flags.DEFINE_integer('layers', 1, 
                      'Number of layers in the RNN.')
  flags.DEFINE_integer('unroll_steps', 15, 
                      'Unrolled network depth.')
  flags.DEFINE_integer('epochs', 10, 
                      'Number of times the dataset is repeated before a checkpoint is stored.')
  flags.DEFINE_integer('max_steps', 20000, 
                      'The learning goes on for this many steps.')
  flags.DEFINE_bool('layer_norm', False, 
                      'Whether to use layer normalization.')
  flags.DEFINE_integer('num_proj', None, 
                      'Output dimensions of projection layer.')
  flags.DEFINE_integer('vocab_size', None, 
                      'Limit vocabulary to this size (only applies to word language models).')
  flags.DEFINE_float('dropout', 0.0,
                      'Set to greater than 0 to add a dropout layer during training. ' +
                      'Parameters will be dropped with this probability (i.e. kept with 1-dropout probability).')
  flags.DEFINE_float('learning_rate', 0.02, 'Learning rate')
  flags.DEFINE_integer('lr_decay_steps', 10000, 
                      'After this many steps, the learning rate drops by a factor of lr_decay_rate.')
  flags.DEFINE_float('lr_decay_rate', 0.6,
                      'After lr_decay_steps steps, the learning rate drops by this factor.')
  flags.DEFINE_float('clip_gradients', None,
                      'Clips gradients to this value.')
  
  flags.DEFINE_string('optimizer', 'Adam', 'Optimizer to use during training')

  flags.DEFINE_enum('language_model', 'word', ['char', 'word'], 'Whether to use a character-level or word-level model.')

  FLAGS = flags.FLAGS
  
  job_name = FLAGS.data_prefix \
    + ('-fp16' if FLAGS.use_fp16 else '') \
    + ('-%d-states' % FLAGS.lstm_state_size) \
    + ('-%d-layers' % FLAGS.layers) \
    + (('-%d-embed' % FLAGS.embedding_size) if FLAGS.embedding_size else '') \
    + ('-norm' if FLAGS.layer_norm else '') \
    + (('-%d-proj' % FLAGS.num_proj) if FLAGS.num_proj else '') \
    + (('-' + FLAGS.language_model) if FLAGS.language_model != 'char' else '') \
    + (('-%dvocab' % FLAGS.vocab_size) if FLAGS.vocab_size else '') 

  logger.info('Job name: ' + job_name)

  # TODO: replace all underscores by dashes?
  setattr(FLAGS, 'job-dir', getattr(FLAGS, 'job-dir').replace('%JOB%', job_name))

  FLAGS.summary_dir = FLAGS.summary_dir.replace('%JOB%', job_name)

  return FLAGS