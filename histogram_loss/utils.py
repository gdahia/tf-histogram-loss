import tensorflow as tf


def flat_strict_upper(mat: tf.Tensor) -> tf.Tensor:
  """
  """
  with tf.name_scope('flat_strict_upper'):
    mask = tf.ones_like(mat, dtype=tf.int32)
    eye = tf.matrix_band_part(mask, 0, 0)
    mask = tf.matrix_band_part(mask, 0, -1) - eye
    mask = tf.cast(mask, dtype=tf.bool)
    return tf.boolean_mask(mat, mask)
