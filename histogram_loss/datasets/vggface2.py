from typing import Tuple

import os
import tensorflow as tf


def decode_and_preprocess(path: str) -> Tuple[tf.Tensor, tf.Tensor]:
  """Decodes and preprocesses the image in the given path with `tf`.

  Args:
    path: string with the path to a `.jpg` image.

  Returns:
    a tuple `(image, label)` with the preprocessed image and its corresponding label.
  """
  # retrieve label
  filename = tf.string_split([path], os.path.sep)
  filename = tf.sparse_tensor_to_dense(filename, default_value='')
  str_label = tf.strings.substr(filename[:, -2], 1, 6)
  label = tf.strings.to_number(str_label, out_type=tf.int32)
  label = tf.squeeze(label)

  # retrieve image
  image_string = tf.read_file(path)
  image = tf.image.decode_jpeg(image_string, channels=3)
  image = tf.image.resize_images(image, [256, 256])
  image = tf.cast(image, dtype=tf.float32)
  image = image / 255

  return image, label


def load(paths_path: str, preffix: str = None) -> tf.data.Dataset:
  """Loads a VGGFace2 datset subset.

  Loads either the training or test subsets of VGGFace2 dataset, depending on the given path.

  Args:
    paths_path: the path to the list containing the image paths to be loaded, e.g. `train_list.txt` for the training subset and `test_list.txt` for the test subset.
    preffix: a string to be preffixed to every path in the paths list.

  Returns:
    a `tf.data.Dataset` with the specified VGGFace2 dataset subset.
  """
  dataset = tf.data.TextLineDataset(paths_path)
  if preffix is not None:
    dataset = dataset.map(lambda path: preffix + path)
  dataset = dataset.map(decode_and_preprocess)

  return dataset
