from abc import ABC, abstractmethod
import string
import re 
import tensorflow as tf
import gzip
import logging

logger = logging.getLogger('language-model')

class LanguageModel(ABC):
  """ Determines whether we are doing character-level or word-level prediction. A "class" is the 
  range of values we tokenize a string into, so the set of valid characters or the set of all
  known words. This class is abstract and intended for subclassing. """
  num_classes = 0 
  
  """ "End of String" token. Used to separate lines. """ 
  EOS = ''


  @abstractmethod
  def tokenize_str(self, str):
    pass

  @abstractmethod
  def id_of_token(self, token):
    """ Gets the ID of a token, e.g. `id_of_token('a') == 1` on a character level model. 
    The inverse of `token`. """
    pass

  @abstractmethod
  def token(self, token_id):
    """ Gets a token by ID, e.g. `token(1) == 'a'` on a character level model. 
    The inverse of `id_of_token`. """
    pass

  @abstractmethod
  def append(self, str, token):
    pass

class CharacterLanguageModel(LanguageModel):
  """ Language model to use for character-level predictions (as opposed to word/dictionary based). """

  def __init__(self):
    self.EOS = '|'
    # classes to predict, i.e. a list of words or characters
    self.CHARS = list(string.ascii_lowercase + ' \'\".,:;!?-+*/()[]{}0123456789#$%' + self.EOS)
    self.CHAR_TO_ID = dict(zip(self.CHARS, range(len(self.CHARS))))

    # "unknown" token
    self.UNK = '#'
    self.UNK_ID = self.CHAR_TO_ID[self.UNK]

    self.CHAR_AS_INT_TO_ID = dict(zip(map(lambda c: ord(c), self.CHARS), range(len(self.CHARS))))
    self.num_classes = len(self.CHARS)
    self.unk_chars_encountered = set()

  def token(self, token_id):
    return self.CHARS[token_id]

  def id_of_token(self, token):
    return self.CHAR_TO_ID[token]

  def char_to_id(self, char):
    if char in self.CHAR_TO_ID:
      return self.CHAR_TO_ID[char] 
    else:
      if not char in self.unk_chars_encountered:
        print('Unknown char ', char)

        if len(self.unk_chars_encountered) > 100:
          raise Exception('There is binary data in the text')

        self.unk_chars_encountered.add(char)

      return self.UNK_ID

  def tokenize_str(self, str):
    if type(str) == bytes:
      str = str.decode('utf-8')

    return [self.char_to_id(ch) for ch in str.lower()]

  def append(self, str, token):
    return str + token


class WordLanguageModel(LanguageModel):
  """ Language model to use for word-level prediction. """

  def __init__(self, vocab_file, max_vocab_size=None):
    self.EOS = '<eos>'
    self.UNK = '<unk>'
    self.words = []
    self.word_to_id = {}
    self.vocab_file = vocab_file
    self.max_vocab_size = max_vocab_size
    
    self._read_vocab()

    self.UNK_ID = self.id_of_token(self.UNK)

  def id_of_token(self, token):
    return self.word_to_id[token] if token in self.word_to_id else self.UNK_ID

  def token(self, token_id):
    return self.words[token_id]

  def tokenize_str(self, str):
    if type(str) == bytes:
      str = str.decode('utf-8')

    str = re.sub(r'([0-9]*)\.([0-9]+)', '\\1_\\2', str)
    str = re.sub(r'([0-9])+\,([0-9][0-9][0-9])', '\\1_\\2', str)

    # newlines to tilde to save them from being splitted on later
    str = str.replace(' // ', ' ~ ')

    words_and_spaces = re.split('([ .;",:!?(){}/])', str)

    # tilde back to newlines
    words_and_spaces = [ '//' if word == '~' else word for word in words_and_spaces ]

    words = filter(lambda x: x.strip() != '', words_and_spaces)

    return [self.id_of_token(word) for word in words]

  def _read_file_contents(self, filename):
    # assuming all data on google cloud storage are gziped
    is_gziped = filename.startswith('gs://')

    with tf.gfile.GFile(filename, "rb" if is_gziped else "r") as file:
      if is_gziped:
        gz_file = gzip.GzipFile(fileobj=file)
        return gz_file.read().decode('utf-8')
      else:
        return file.read()

  def _read_vocab(self):
    words = self._read_file_contents(self.vocab_file).split('\n')

    if not self.EOS in words:
      words = [self.EOS] + words

    if not self.UNK in words:
      words = [self.UNK] + words
    
    if self.max_vocab_size and len(words) > self.max_vocab_size:
      logger.warn('%d words in vocabulary. Restricting to %d' % (len(words), self.max_vocab_size))

      words = words[0:self.max_vocab_size]
    else:
      logger.info('%d words in vocabulary.' % len(words))

    self.words = words
    self.word_to_id = dict(zip(words, range(len(words))))
    self.num_classes = len(words)

  def append(self, str, token):
    if token == '//':
      token_str = '\n'
    elif token == '<bullet>':
      token_str = '*'
    else:
      token_str = token

    if token_str in '\'.,:;!?)]}':
      return str + token_str
    else: 
      return str + ' ' + token_str
