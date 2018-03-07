import numpy as np
import tensorflow as tf
from tensorflow.contrib.learn import ModeKeys

class SlicerUnderflowException(Exception):
  pass

class EqualLengthBatchSlicer():
  """
  Takes batches that are arrays of token IDs and 
  returns new batches where each item has the same length.
  item that are longer will be deferred until the next batch.
  """
  
  def __init__(self, hyperparameters):
    """
    Args:
      * `min_length` The minimum length of a batch entry.
    """
    self.batch_size = hyperparameters.batch_size
    self.min_length = hyperparameters.unroll_steps + 1
    self.EOS_ID = hyperparameters.language_model.id_of_token(hyperparameters.language_model.EOS)
    self.buffer = [[self.EOS_ID]] * self.batch_size

  def add_batch(self, batch):
    """
    Given a batch of arrays of token (word/character) IDs, will add these to whatever
    data has not yet been returned and return a batch of ID arrays that are
    all the same length. Any arrays that are longer will be buffered and returned
    at the beginning of next batch.

    Note that batches overlap by `min_length-1` token in order that, when sliced
    into `min_length` long slices, they do not skip any positions.

    Args:
      * `batch` An array of number arrays of length `batch_size` or shorter.

    Returns:
      * An array of length `batch_size` where all items are number arrays that
        are of equal length and least `min_length` long. 
        
    Raises:
      * `SlicerUnderflowException` If such an array cannot be constructed because
        there is not enough buffered data and the input batch is also too short.
    """
    assert len(batch) <= self.batch_size

    # add an end of string token to each item
    batch = [token_array + [self.EOS_ID] for token_array in batch]

    # pad the batch to always be exactly `batch_size` long
    if len(batch) < self.batch_size:
      batch = batch + [[]] * (self.batch_size - len(batch))

    # buffer = buffer + batch
    self.buffer = [a + b for (a, b) in zip(self.buffer, batch)]

    min_len = min([len(token_array) for token_array in self.buffer])

    if min_len < self.min_length:
      raise SlicerUnderflowException()

    overlap = self.min_length - 1

    # new batch is everything up to the length of the shortest token array
    this_batch = [s[0:min_len] for s in self.buffer]

    # the buffer is everything after that plus the overlap we want between batches
    self.buffer = [s[max(min_len - overlap, 0):] for s in self.buffer]

    return this_batch

def slice(token_ids, slice_length, language_model):
  """
  Slices a string into slices of length `slice_length`. Slices overlap by one character.
  e.g. `string_to_slices('abcde', 3) == ['abc', 'cde']` (in the case of character-level training)
  """
  def get_slice(start_index): 
    return token_ids[start_index:start_index + slice_length]

  slice_count = len(token_ids) - slice_length + 1

  if slice_count > 0: 
    return [get_slice(start_index) for start_index in range(0, slice_count, slice_length-1)]
  else:
    return []

def string_batch_to_slices_batch(batch, hyperparameters):
  """
  Given a batch of `batch_size` strings, returns a batch of training data. The training data is a `tf.data.Dataset`
  containing a tuple of of input and target tensors. Both have shape `[batch_size, slice_count, unroll_steps]` where 
  `slice_count` is the number of slices produced. A slice is `num_length` characters from the input string, converted
  into numbers (corresponding to class IDs).
   
  The number of slices depends on the length of the strings and of any previously buffered data. It is, however,
  guaranteed that all batches returned have the same slice count.

  Passing the return data to `tf.data.Dataset.interleave` will produce the final training data.

  Note that it's extremely important that the batches have the same number of slices. Subsequent slices from the same
  input string need to maintain the same position in a batch, as the LSTM otherwise does not see the current sequence
  and has not chance to learn it. 
  
  That can only be guaranteed by making sure the number slices is the same in all datasets before `interleave` is called.

  Note that is also means that we may drop some of the training data at the end once one position in the batch does 
  not have any more data.
  """
  unroll_steps = hyperparameters.unroll_steps
  batch_size = hyperparameters.batch_size
  slicer = EqualLengthBatchSlicer(hyperparameters)

  def handle_batch(batch):
    try:
      equal_length_batch = slicer.add_batch(
        [hyperparameters.language_model.tokenize_str(str) for str in batch])

      if len(equal_length_batch) > 0:
        assert len(equal_length_batch) == batch_size
        assert type(equal_length_batch[0]) == list
        assert type(equal_length_batch[0][0]) == int

      # replace each token ID array with an array of slices
      slices = [slice(token_ids, 
        slice_length=unroll_steps+1, language_model=hyperparameters.language_model) 
          for token_ids in equal_length_batch]

      # return a two-dimensional numpy array of shape [slice_count, unroll_steps+1]
      return np.array(slices, dtype=np.int64)
    except SlicerUnderflowException:
      return np.zeros((0, 0, unroll_steps+1), np.int64)

  # `slices`` has shape [batch_size, slice_count, unroll_steps+1]. 
  # `batch_size` may be zero if we had a slicer underflow.
  slices = tf.py_func(handle_batch, [batch], tf.int64)

  with tf.control_dependencies([
      tf.assert_rank(slices, 3),
      tf.assert_equal(tf.shape(slices)[2], unroll_steps+1)
    ]):
    # Inputs to the training are the first unroll_steps token IDs of slice 
    # (i.e. all except the last token ID)
    input = slices[:, :, :-1]

    # Labels for the training are the last unroll_steps token IDs of slice 
    # (i.e. all except the first token ID)
    labels = slices[:, :, 1:]

    # We need to explicitly reshape the tensors for the dataset API to 
    # understand the shape.
    shape = tf.shape(slices)
    slice_count = shape[1]
    # will be batch_size or 0
    this_batch_size = shape[0]

    def make_shape_explicit(tensor):
      return tf.reshape(tensor, [this_batch_size, slice_count, unroll_steps])

    return tf.data.Dataset.from_tensor_slices((make_shape_explicit(input), make_shape_explicit(labels)))

# read data and convert to needed format
def read_dataset(filename, hyperparameters, mode=ModeKeys.TRAIN):
  def _input_fn(*args):
    num_epochs = hyperparameters.epochs if mode == ModeKeys.TRAIN else 1
    batch_size = hyperparameters.batch_size

    dataset = tf.data.TextLineDataset(filename, compression_type='GZIP' if filename.startswith('gs://') else None)

    if mode == ModeKeys.TRAIN:
      dataset = dataset.shuffle(buffer_size=hyperparameters.batch_size * 100)
    
    # batch into `batch_size` lines per batch
    dataset = dataset.batch(batch_size)

    # map to `batch_size` tuples of input and target slice arrays per batch. 
    # each slice array is a the same length within a batch
    # (but not between batches) and contains slices of shape [num_step_steps]
    dataset = dataset.flat_map(lambda batch: string_batch_to_slices_batch(batch, hyperparameters))

    # now, flatten the structure so we get a list of input/target tuples
    # tuples are interleaved in the order 1, 2, 3... `batch_size-1`, 1, 2, 3... etc
    # this makes sure each LSTM sees the samples in the order occurred in the input string.
    dataset = dataset.interleave(lambda *x: tf.data.Dataset.from_tensor_slices(x), batch_size, 1)

    # drop any samples that don't form a complete batch
    dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))
    dataset = dataset.repeat(num_epochs)

    return dataset

  return _input_fn
