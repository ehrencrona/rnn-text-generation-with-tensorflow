from trainer.language_model import WordLanguageModel, CharacterLanguageModel
from os import path

class Hyperparameters():
  """ Gathers all hyperparameters of the model. """

  def __init__(self, FLAGS):
    self.lstm_state_size = 5
    self.batch_size = 3
    self.unroll_steps = 3
    self.epochs = 10
    self.dropout = 0.0
    self.optimizer = 'Adam'
    self.layer_norm = False
    self.num_proj = None
    self.learning_rate = 0.5
    self.lr_decay_steps = 10000
    self.lr_decay_rate = 0.8
    self.clip_gradients = 5
    self.layers = 1
    self.embedding_size = None

    for param in dir(FLAGS):
      if param in dir(self):
        setattr(self, param, getattr(FLAGS, param))

    if FLAGS.language_model == 'char':
      self.language_model = CharacterLanguageModel() 
    else:
      vocab_file = path.join(FLAGS.data_dir, FLAGS.data_prefix + '.vocab.txt')

      self.language_model = WordLanguageModel(vocab_file, FLAGS.vocab_size)
