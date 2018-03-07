import tensorflow as tf

def debug_print(op):
  """ Utility for printing a tensor's value at runtime. """
  return tf.Print(op, [op.name, tf.shape(op), op], first_n=100, summarize=30)

def print_dataset(name):
  """ Utility for printing the data in a dataset. """
  def really_print(*x):
    def print(op):
      return tf.Print(op, [name, x[0], x[1]], first_n=56, summarize=30)

    return (print(x[0]), x[1])

  return really_print
