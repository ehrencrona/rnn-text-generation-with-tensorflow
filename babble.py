
from trainer.reader import _read_vocab, UNK
import tensorflow as tf
import random
from trainer.task import PTBModel, PTBPredictInput, PTBInput, get_config

def read_word_to_id(vocab_size):
  return _read_vocab(max_vocab_size=vocab_size)



config = get_config()
config.batch_size = 1
config.num_steps = 1

vocab = read_word_to_id(config.vocab_size)

chars = vocab['chars']
word_to_id = vocab['char_to_id']

initializer = tf.random_uniform_initializer(-config.init_scale,
                                              config.init_scale)

def get_word(id):
  return chars[id]

def get_id_of_word(word):
  return word_to_id[word]

def vector_to_word_id(embeddings, embedded_vector):
  print('embeddings', embeddings)
  print('embedded_vector', embedded_vector)
  
  emb_distances = tf.matmul(
      tf.nn.l2_normalize(embeddings, axis=1),
      tf.nn.l2_normalize(embedded_vector, axis=1),
      transpose_b=True)

  token_ids = tf.argmax(emb_distances, axis=0)

  return token_ids[0]
  
last_word_id = get_id_of_word('A')

valid_data = [ last_word_id ]

with tf.name_scope("Train"):
  ptb_input = PTBPredictInput(config=config, data=valid_data, name="Input")
  
  with tf.variable_scope("Model", reuse=None, initializer=initializer):
    m = PTBModel(is_training=True, is_predict=True, config=config, input_=ptb_input)

    with tf.Session() as sess:
      print(tf.global_variables())
      coord = tf.train.Coordinator()
      threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    
      PATH = '/projects/onthejob/training/data/character-level-noembed/'
      CKPT = 'model.ckpt-357570'

      saver = tf.train.Saver()
      saver.restore(sess, PATH + CKPT)

      vocab_size = 101 # len(word_to_id)

      logits = tf.reshape(m._logits, [ vocab_size ])
            
      sentence = ''
      lstm_state = None

      print(tf.global_variables())

      for i in range(0, 1000):
        feed_dict = {
          ptb_input.input_data: [[last_word_id]]
        }

        if lstm_state:
          feed_dict[m._initial_state] = lstm_state

        v = sess.run({
          'logits': logits,
          'word_id': tf.argmax(logits, axis=0),
          'top_word_ids': tf.nn.top_k(logits, k=10),
          'sampled_word': tf.multinomial(tf.reshape(m._logits, [ 1, vocab_size ]), num_samples=1),
          'input': ptb_input.input_data,
          'lstm_state': m.lstm_state,        
        }, feed_dict=feed_dict)

        lstm_state = v['lstm_state']
        top_word_ids = v['top_word_ids'].indices
        new_word_id = v['sampled_word'][0][0]
        
        last_word_id = new_word_id

        sentence = sentence + get_word(new_word_id)

      print(sentence.replace(' // ', '\n').replace('<bullet>', ' * '))