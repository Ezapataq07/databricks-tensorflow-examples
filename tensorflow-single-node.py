# Databricks notebook source
# MAGIC %md ## TensorFlow tutorial - MNIST For ML Beginners
# MAGIC
# MAGIC This notebook demonstrates how to use TensorFlow on the Spark driver node to fit a neural network on MNIST handwritten digit recognition data.
# MAGIC
# MAGIC Prerequisites:
# MAGIC * A GPU-enabled cluster.
# MAGIC * TensorFlow 1.15 or 2.x with GPU support installed manually.
# MAGIC
# MAGIC The content of this notebook is [adapted from TensorFlow project](https://www.tensorflow.org/tutorials/quickstart/beginner) under [Apache 2.0 license](https://github.com/tensorflow/tensorflow/blob/master/LICENSE) with slight modification to run on Databricks. Thanks to the developers of TensorFlow for this example!

# COMMAND ----------

import tensorflow as tf
from tensorflow.keras import models, layers, datasets
import datetime, uuid

# COMMAND ----------

# Verify that we're using TensorFlow 2.x or 1.15
assert tf.__version__.startswith("2.") or tf.__version__.startswith("1.15")

# COMMAND ----------

tf.__version__

# COMMAND ----------

# MAGIC %md Load the data (this step may take a while)

# COMMAND ----------

(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# COMMAND ----------

# MAGIC %md Define the model

# COMMAND ----------

model = models.Sequential([
  layers.Flatten(input_shape=(28, 28)),
  layers.Dense(10, activation='softmax')
])

# COMMAND ----------

# MAGIC %md Define loss and optimizer

# COMMAND ----------

model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.5),
              loss=tf.keras.losses.sparse_categorical_crossentropy,
              metrics=['accuracy'])

# COMMAND ----------

# MAGIC %md
# MAGIC Start TensorBoard so you can monitor training progress.

# COMMAND ----------

# Define a user unique directory in DBFS
try:
  username = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().apply('user')
except:
  username = str(uuid.uuid1()).replace("-", "")
experiment_log_dir = "/dbfs/user/{}/tensorboard_log_dir/".format(username)

# COMMAND ----------

# MAGIC %load_ext tensorboard

# COMMAND ----------

# MAGIC %tensorboard --logdir $experiment_log_dir

# COMMAND ----------

type(x_train)

# COMMAND ----------

# MAGIC %md Train the model in batches

# COMMAND ----------

run_log_dir = experiment_log_dir + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=run_log_dir, histogram_freq=1)

model.fit(x_train, y_train, epochs=5, batch_size=64, callbacks=[tensorboard_callback])

# COMMAND ----------

# MAGIC %md Test the trained model. The final accuracy is reported at the bottom. You can compare it with the accuracy reported by the other frameworks!

# COMMAND ----------

model.evaluate(x_test,  y_test, verbose=2)

# COMMAND ----------

# MAGIC %md
# MAGIC TensorBoard stays active after your training is finished so you can view a summary of the process. Detach your notebook to stop the Tensorboard.
# MAGIC
# MAGIC (Optional) Remove your log files from DBFS.

# COMMAND ----------

dbutils.fs.rm(experiment_log_dir.replace("/dbfs",""), recurse=True)
