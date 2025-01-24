import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K
import os
import bz2
from glob import glob

def data_generator(files):
  for file in files:
    for chunk in pd.read_csv(file, compression='bz2', header=None, sep=r"\s+", chunksize=1000):
      X = tf.stack([tf.convert_to_tensor(chunk.iloc[:, 0].values*chunk.iloc[:, 1].values, dtype=tf.float32), 
                    tf.convert_to_tensor(chunk.iloc[:, 0].values*chunk.iloc[:, 2].values, dtype=tf.float32), 
                    tf.convert_to_tensor(chunk.iloc[:, 3].values, dtype=tf.float32)])
      X=tf.transpose(X)
      X=tf.expand_dims(X, axis=1)
      y_abs = []
      y_p = []
      y_qty = []
      for i in range (3, chunk.shape[1], 3):
        y_abs.append(tf.convert_to_tensor(chunk.iloc[:, i].values, dtype=tf.float32))
        y_p.append(tf.convert_to_tensor(chunk.iloc[:, i+1].values, dtype=tf.float32))
        y_qty.append(tf.convert_to_tensor(chunk.iloc[:, i+2].values, dtype=tf.float32))

      y_abs_tensor = tf.stack(y_abs, axis=1)
      y_p_tensor = tf.stack(y_p, axis=1)
      y_qty_tensor = tf.stack(y_qty, axis=1)

      y=tf.stack([y_abs_tensor, y_p_tensor, y_qty_tensor], axis=-1)

      for i in range(len(X)):
        yield X[i], y[i]

def find_files(directory):
  return glob(os.path.join(directory, '*.bz2'))

directory='./'
files=find_files(directory)

def filter_dataset(x, y):
  return tf.reduce_all(tf.math.is_finite(x)) and tf.reduce_all(tf.math.is_finite(y))

def create_tf_dataset(files):
  dataset=tf.data.Dataset.from_generator(lambda: data_generator(files), 
    output_signature=(tf.TensorSpec(shape=(1, 3), dtype=tf.float32),
                      tf.TensorSpec(shape=(100, 3), dtype=tf.float32)))
  dataset=dataset.filter(filter_dataset)
  return dataset

batch_size=32
dataset=create_tf_dataset(files)
dataset=dataset.batch(batch_size)

def mae_loss(y_true, y_pred):
  qty=y_true[:, :, 2]
  mae = tf.reduce_mean(tf.abs(y_true[:, :, :2] - y_pred), axis=-1)
  loss=mae*qty
  return tf.reduce_mean(loss)

model = keras.Sequential()
model.add(layers.Input(shape=(1,3)))
model.add(layers.LSTM(50, activation='relu'))
model.add(layers.RepeatVector(100))
model.add(layers.LSTM(50, activation='relu', return_sequences=True))
model.add(layers.Dense(2))
model.summary()

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-6), loss=mae_loss)

history=model.fit(dataset, epochs=10)

print(history.history['loss'])
print(model.evaluate())
quit()
model.save('blabla.keras')

