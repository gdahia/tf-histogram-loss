from typing import List, Tuple

import tensorflow as tf

from histogram_loss.datasets.utils import create_tf_dataset


def separate_paths_by_label(paths: List[str]) -> List[List[str]]:
  """
  """


def preprocess(path: str) -> Tuple[tf.Tensor, tf.Tensor]:
  """
  """
  # retrieve label
  filename = tf.string_split(path, '/')[-1]
  wo_extension = tf.string_split(filename, '.')[-1]
  str_label = tf.strings.substr(wo_extension, 1, 6)
  label = tf.strings.to_number(str_label, out_type=tf.int32)

  # retrieve image
  image_string = tf.read_file(path)
  image = tf.image.decode_jpeg(image_string)
  # image = tf.image.resize_images(image, [28, 28])  # hardcode proper size

  return image, label


def load_vggface2_dataset(paths_path: str,
                          preffix_path: str = None) -> tf.train.Dataset:
  """
  """
  # read paths from file and preffix them
  paths = [path.strip() for path in open(paths_path, 'r')]
  paths = sorted(paths)
  if preffix_path is not None:
    paths = [preffix_path + path for path in paths]

  paths_by_label = separate_paths_by_label(paths)

  return create_tf_dataset(paths_by_label, preprocess)
