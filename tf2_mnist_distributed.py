# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import, unicode_literals, division, print_function

import argparse
import json
import logging
import os
import sys
import numpy as np

# tensorflow >= 1.15
import tensorflow_datasets as tfds
import tensorflow as tf
tfds.disable_progress_bar()

# CLI
# parser = argparse.ArgumentParser(description='Train model')
# parser.add_argument('--batch_size', type=int, default=32,

# In general, the largest batch size that fits the GPU memory is advisable.
BATCH_SIZE = 128
BUFFER_SIZE = 10000
LEARNING_RATE = 1e-4


def input_fn(features, labels, batch_size, mode):
    """Input function.

    Args:
      features: (numpy.array) Training or eval data.
      labels: (numpy.array) Labels for training or eval data.
      batch_size: (int)
      mode: tf.estimator.ModeKeys mode

    Returns:
      A tf.estimator.
    """
    # Default settings for training.
    if labels is None:
        inputs = features
    else:
        # Change numpy array shape.
        inputs = (features, labels)
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices(inputs)
    if mode == tf.estimator.ModeKeys.TRAIN:
        dataset = dataset.shuffle(1000).repeat().batch(batch_size)
        dataset = dataset.prefetch(100)
    if mode in (tf.estimator.ModeKeys.EVAL, tf.estimator.ModeKeys.PREDICT):
        dataset = dataset.batch(batch_size)
    return dataset

def model_fn(features, labels, mode):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10)
    ])

    logits = model(features, training=False)

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {'logits': logits}
        return tf.estimator.EstimatorSpec(labels=labels, predictions=predictions)

    optimizer = tf.compat.v2.optimizers.SGD (learning_rate=LEARNING_RATE)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction=tf.keras.losses.Reduction.NONE)(labels, logits)
    loss = tf.reduce_sum(loss) * (1. / BATCH_SIZE)
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss)

    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=optimizer.minimize(
            loss, tf.compat.v1.train.get_or_create_global_step()))

def create_model(model_dir, config, learning_rate):
    """Creates a Keras Sequential model with layers.

        Args:
          model_dir: (str) file path where training files will be written.
          config: (tf.estimator.RunConfig) Configuration options to save model.
          learning_rate: (int) Learning rate.

        Returns:
          A keras.Model
        """
    l = tf.keras.layers
    model = tf.keras.Sequential(
        [
            l.Reshape(input_shape=(28 * 28,), target_shape=(28, 28, 1)),

            l.Conv2D(filters=6, kernel_size=3, padding='same',
                     use_bias=False),
            # no bias necessary before batch norm
            l.BatchNormalization(scale=False, center=True),
            # no batch norm scaling necessary before "relu"
            l.Activation('relu'),  # activation after batch norm

            l.Conv2D(filters=12, kernel_size=6, padding='same',
                     use_bias=False,
                     strides=2),
            l.BatchNormalization(scale=False, center=True),
            l.Activation('relu'),

            l.Conv2D(filters=24, kernel_size=6, padding='same',
                     use_bias=False,
                     strides=2),
            l.BatchNormalization(scale=False, center=True),
            l.Activation('relu'),

            l.Flatten(),
            l.Dense(200, use_bias=False),
            l.BatchNormalization(scale=False, center=True),
            l.Activation('relu'),
            l.Dropout(0.5),  # Dropout on dense layer only

            l.Dense(10, activation='softmax')
        ])
    # Compile model with learning parameters.
    optimizer = tf.compat.v2.optimizers.SGD (learning_rate=LEARNING_RATE)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])
    tf.keras.backend.set_learning_phase(True)
    model.summary()
    estimator = tf.keras.estimator.model_to_estimator(
        keras_model=model, model_dir=model_dir, config=config)
    return estimator

def serving_input_fn():
    """Defines the features to be passed to the model during inference.

    Expects already tokenized and padded representation of sentences

    Returns:
      A tf.estimator.export.ServingInputReceiver
    """
    feature_placeholder = tf.placeholder(tf.float32, [None, 28 * 28])
    features = feature_placeholder
    return tf.estimator.export.TensorServingInputReceiver(features,
                                                          feature_placeholder)

# def _get_session_config_from_env_var():
#     """Returns a tf.ConfigProto instance with appropriate device_filters set."""
#     tf_config = json.loads(os.environ.get('TF_CONFIG', '{}'))

#     # GPU limit: TensorFlow by default allocates all GPU memory:
#     # If multiple workers run in same host you may see OOM errors:
#     # Use as workaround if not using Hadoop 3.1
#     # Change percentage accordingly:
#     # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.25)

#     if (tf_config and 'task' in tf_config and 'type' in tf_config['task'] and
#         'index' in tf_config['task']):
#         # Master should only communicate with itself and ps.
#         if tf_config['task']['type'] == 'master':
#             return tf.compat.v1.ConfigProto(device_filters=['/job:ps', '/job:master'])
#         # Worker should only communicate with itself and ps.
#         elif tf_config['task']['type'] == 'worker':
#             return tf.compat.v1.ConfigProto(#gpu_options=gpu_options,
#                                   device_filters=[
#                                       '/job:ps',
#                                       '/job:worker/task:%d' % tf_config['task'][
#                                           'index']
#                                   ])
#     return None

def main():
    logging.getLogger().setLevel(logging.INFO)

    strategy = tf.distribute.experimental.ParameterServerStrategy()

    (train_images, train_labels), (
        test_images, test_labels) = tf.keras.datasets.mnist.load_data()

    # Scale values to a range of 0 to 1.
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    # Shape numpy array.
    train_labels = np.asarray(train_labels).astype('int').reshape((-1, 1))
    test_labels = np.asarray(test_labels).astype('int').reshape((-1, 1))

    # Define training steps.
    train_steps = len(train_images) / BATCH_SIZE

    config = tf.estimator.RunConfig(train_distribute=strategy)


    classifier = create_model(
        model_dir='/tmp/mode',
        config=config,
        learning_rate=LEARNING_RATE)


    # Create TrainSpec.
    train_spec = tf.estimator.TrainSpec(
        input_fn=lambda: input_fn(
            train_images,
            train_labels,
            BATCH_SIZE,
            mode=tf.estimator.ModeKeys.TRAIN),
            max_steps=train_steps)

    # Create EvalSpec.
    exporter = tf.estimator.FinalExporter('exporter', serving_input_fn)
    eval_spec = tf.estimator.EvalSpec(
        input_fn=lambda: input_fn(
            test_images,
            test_labels,
            BATCH_SIZE,
            mode=tf.estimator.ModeKeys.EVAL),
        steps=None,
        name='mnist-eval',
        exporters=[exporter],
        start_delay_secs=10,
        throttle_secs=10)

    tf.estimator.train_and_evaluate(
        classifier,
        train_spec=train_spec,
        eval_spec=eval_spec
    )

if __name__ == '__main__':
    main()


