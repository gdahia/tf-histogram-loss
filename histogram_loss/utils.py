import tensorflow as tf


def flat_strict_upper(mat: tf.Tensor) -> tf.Tensor:
  """Converts the strictly upper triangular part of a matrix into an array.

  Args:
    mat: a `tf.Tensor` matrix.

  Returns:
    The strictly upper triangular part of `mat` in a `tf.Tensor` array.
  """
  with tf.name_scope('flat_strict_upper'):
    partitions = tf.ones_like(mat, dtype=tf.int32)
    eye = tf.matrix_band_part(partitions, 0, 0)
    partitions = tf.matrix_band_part(partitions, 0, -1) - eye
    return tf.dynamic_partition(mat, partitions, 2)[1]
