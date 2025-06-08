# Databricks notebook source
# MAGIC %md
# MAGIC # Distributed Training with TensorFlow 2
# MAGIC
# MAGIC [tensorflow.distribute.Strategy](https://www.tensorflow.org/guide/distributed_training) is a TensorFlow API to distribute training across multiple GPUs or multiple machines. The `spark-tensorflow-distributor` package helps you launch distributed training tasks using a Spark job in barrier mode. Users only need to provide a `train()` function that runs the single-node training code on a GPU or worker node and the package handles all the configurations for you.
# MAGIC
# MAGIC This notebook demonstrates how to use [MirroredStrategyRunner](https://github.com/tensorflow/ecosystem/blob/master/spark/spark-tensorflow-distributor/spark_tensorflow_distributor/mirrored_strategy_runner.py) in the `spark-tensorflow-distributor` package to do distributed training. It also shows how to use your own custom strategy. The example is adapted from https://www.tensorflow.org/tutorials/distribute/multi_worker_with_keras.
# MAGIC
# MAGIC ### Requirements
# MAGIC * Databricks Runtime ML 7.0 and above
# MAGIC * (Recommended) GPU instances

# COMMAND ----------

import tensorflow as tf
NUM_WORKERS = 2
# Assume the driver node and worker nodes have the same instance type.
TOTAL_NUM_GPUS = len(tf.config.list_logical_devices('GPU')) * NUM_WORKERS
USE_GPU = TOTAL_NUM_GPUS > 0

# COMMAND ----------

# MAGIC %md ## Write single-node code in the train() function
# MAGIC
# MAGIC When you wrap the single-node code in the `train()` function, Databricks recommends you include all the `import` statements inside the `train()` function to avoid library pickling issues. It is also helpful to set the [data auto-shard policy](https://www.tensorflow.org/api_docs/python/tf/data/experimental/DistributeOptions) to DATA.

# COMMAND ----------

def train(batch_size):
  import tensorflow as tf
  import numpy as np
  import uuid

  BUFFER_SIZE = 10000

  def make_datasets(batch_size):
    (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()

    # The `x` arrays are in uint8 and have values in the [0, 255] range.
    # You need to convert them to float32 with values in the [0, 1] range.
    x_train = x_train / np.float32(255)
    y_train = y_train.astype(np.int64)
    train_dataset = tf.data.Dataset.from_tensor_slices(
      (x_train, y_train)).shuffle(BUFFER_SIZE).repeat(2).batch(batch_size)
    return train_dataset

  def build_and_compile_cnn_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(128, 3, activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(),
        # tf.keras.layers.Conv2D(256, 3, activation='relu'),
        # tf.keras.layers.MaxPooling2D(),
        # tf.keras.layers.Conv2D(512, 3, activation='relu'),
        # tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        # tf.keras.layers.Dense(512, activation='relu'),
        # tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
        ])
    model.compile(
      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
      metrics=['accuracy'],
    )
    return model
  
  train_datasets = make_datasets(batch_size)
  multi_worker_model = build_and_compile_cnn_model()

  # Specify the data auto-shard policy: DATA
  options = tf.data.Options()
  options.experimental_distribute.auto_shard_policy = \
      tf.data.experimental.AutoShardPolicy.DATA
  train_datasets = train_datasets.with_options(options)
  
  multi_worker_model.fit(x=train_datasets, epochs=1)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Distributed Training with MirroredStrategyRunner
# MAGIC ### MirroredStrategyRunner: Local mode
# MAGIC
# MAGIC In the local mode, the `train()` function runs on the driver node with all GPUs.

# COMMAND ----------

from spark_tensorflow_distributor import MirroredStrategyRunner

BATCH_SIZE_PER_REPLICA = 512

runner = MirroredStrategyRunner(num_slots=1, local_mode=True, use_gpu=USE_GPU)
runner.run(train, batch_size=BATCH_SIZE_PER_REPLICA)

# COMMAND ----------

# MAGIC %md 
# MAGIC ### MirroredStrategyRunner: Distributed mode
# MAGIC
# MAGIC MirroredStrategyRunner with `local_mode=False` (default) runs the `train()` function on the worker nodes of the Spark cluster.
# MAGIC
# MAGIC Specify the total number of GPUs to use for this run using the parameter, `num_slots=TOTAL_NUM_GPUS` in the `MirroredStrategyRunner` distributor.
# MAGIC
# MAGIC To configure the number of GPUs to use for each Spark task that runs the `train` function, set `spark.task.resource.gpu.amount <num_gpus_per_task>` in the **Spark Config** cell on the cluster page before creating the cluster.
# MAGIC
# MAGIC **Note**: The performance of this run may not be as fast as the single node run because running multiple workers adds overhead. The goal here is to give an example of multi-worker training.
# MAGIC
# MAGIC To reduce the communication overhead, Databricks recommendeds maximizing the GPU ultilzation on each GPU. This is typically done by using the largest batch size that fits into memory. You can do so, by setting the parameter `batch_size` to **the largest batch size that fits into a single GPU * number of slots**.
# MAGIC

# COMMAND ----------

NUM_SLOTS = TOTAL_NUM_GPUS if USE_GPU else 2  # For CPU training, choose a reasonable NUM_SLOTS value
runner = MirroredStrategyRunner(num_slots=NUM_SLOTS, use_gpu=USE_GPU)
runner.run(train, batch_size=BATCH_SIZE_PER_REPLICA*NUM_SLOTS)

# COMMAND ----------

# MAGIC %md ## MirroredStrategyRunner: Use custom strategy
# MAGIC
# MAGIC You can use a custom strategy with the `MirroredStrategyRunner`. You need to construct and use your own `tf.distribute.Strategy` object in the `train()` function and pass `use_custom_strategy=True` to `MirroredStrategyRunner`.
# MAGIC
# MAGIC For example, you can construct a custom strategy which uses `tf.distribute.experimental.MultiWorkerMirroredStrategy`.

# COMMAND ----------

def train_custom_strategy():
  import tensorflow as tf
  
  strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy(
      tf.distribute.experimental.CollectiveCommunication.NCCL)
  
  with strategy.scope(batch_size):
    import uuid

    BUFFER_SIZE = 10000

    def make_datasets():
      (mnist_images, mnist_labels), _ = \
          tf.keras.datasets.mnist.load_data(path=str(uuid.uuid4())+'mnist.npz')

      dataset = tf.data.Dataset.from_tensor_slices((
          tf.cast(mnist_images[..., tf.newaxis] / 255.0, tf.float32),
          tf.cast(mnist_labels, tf.int64))
      )
      dataset = dataset.repeat().shuffle(BUFFER_SIZE).batch(batch_size)
      return dataset

    def build_and_compile_cnn_model():
      model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax'),
      ])
      model.compile(
        loss=tf.keras.losses.sparse_categorical_crossentropy,
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
        metrics=['accuracy'],
      )
      return model

    train_datasets = make_datasets()
    multi_worker_model = build_and_compile_cnn_model()

    # Specify the data auto-shard policy: DATA
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = \
        tf.data.experimental.AutoShardPolicy.DATA
    train_datasets = train_datasets.with_options(options)

    multi_worker_model.fit(x=train_datasets, epochs=3, steps_per_epoch=5)

# COMMAND ----------

# Use the local mode to verify `CollectiveCommunication.NCCL` is printed in the logs
runner = MirroredStrategyRunner(num_slots=1, use_custom_strategy=True, local_mode=True, use_gpu=USE_GPU)
runner.run(train_custom_strategy, batch_size=BATCH_SIZE_PER_REPLICA)
