# tensorflow 2 supported
# using MultiWorkerMirroredStrategy
# Need to install tf-nightly
# https://www.tensorflow.org/tutorials/distribute/multi_worker_with_keras

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow_datasets as tfds
import tensorflow as tf
tfds.disable_progress_bar()

BUFFER_SIZE = 10000
BATCH_SIZE = 64
NUM_WORKERS = 2
GLOBAL_BATCH_SIZE = 64 * NUM_WORKERS
strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()

def make_datasets_unbatched():
  # Scaling MNIST data from (0, 255] to (0., 1.]
  def scale(image, label):
    image = tf.cast(image, tf.float32)
    image /= 255
    return image, label

  datasets, info = tfds.load(name='mnist',
  							data_dir='/tmp/data',
                            with_info=True,
                            as_supervised=True)

  return datasets['train'].map(scale).cache().shuffle(BUFFER_SIZE)

def build_and_compile_cnn_model():
  model = tf.keras.Sequential([
      tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
      tf.keras.layers.MaxPooling2D(),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(64, activation='relu'),
      tf.keras.layers.Dense(10)
  ])
  model.compile(
      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
      metrics=['accuracy'])
  return model


def main():

    # train_datasets = make_datasets_unbatched().batch(BATCH_SIZE)

    with strategy.scope():
        # Creation of dataset, and model building/compiling need to be within 
        # `strategy.scope()`.
        train_datasets = make_datasets_unbatched().batch(GLOBAL_BATCH_SIZE)
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
        train_datasets_no_auto_shard = train_datasets.with_options(options)
        multi_worker_model = build_and_compile_cnn_model()

    # Keras' `model.fit()` trains the model with specified number of epochs and
    # number of steps per epoch. Note that the numbers here are for demonstration
    # purposes only and may not sufficiently produce a model with good quality.
    multi_worker_model.fit(x=train_datasets_no_auto_shard, epochs=3, steps_per_epoch=5)

if __name__ == '__main__':
    main()


