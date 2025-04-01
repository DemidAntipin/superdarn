import numpy as np
import random
import os
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from glob import glob

def find_files(directory):
  return glob(os.path.join(directory, '*.fitacf'))

directory='data_short/'

def filter_dataset(x, y):
  return tf.reduce_all(tf.math.is_finite(x)) & tf.reduce_all(tf.math.is_finite(y[0])) & tf.reduce_all(tf.math.is_finite(y[1]))

def create_tf_dataset():
  batch_size=1
  data_gen_train = DataGenerator(directory+'train/')
  data_gen_val = DataGenerator(directory+'validation/')
  data_gen_test = DataGenerator(directory+'test/')
  train_dataset=tf.data.Dataset.from_generator(lambda: data_gen_train, 
    output_signature=(tf.TensorSpec(shape=(1, 3), dtype=tf.float32),
                      (tf.TensorSpec(shape=(100, 2), dtype=tf.float32),
		      tf.TensorSpec(shape=(100, 2), dtype=tf.float32))))
  train_dataset=train_dataset.filter(filter_dataset).batch(batch_size)

  val_dataset=tf.data.Dataset.from_generator(lambda: data_gen_val,
    output_signature=(tf.TensorSpec(shape=(1, 3), dtype=tf.float32),
                      (tf.TensorSpec(shape=(100, 2), dtype=tf.float32),
		      tf.TensorSpec(shape=(100, 2), dtype=tf.float32))))
  val_dataset=val_dataset.filter(filter_dataset).batch(batch_size)

  test_dataset=tf.data.Dataset.from_generator(lambda: data_gen_test,
    output_signature=(tf.TensorSpec(shape=(1, 3), dtype=tf.float32),
  		      (tf.TensorSpec(shape=(100, 2), dtype=tf.float32),
		      tf.TensorSpec(shape=(100, 2), dtype=tf.float32))))
  test_dataset=test_dataset.filter(filter_dataset).batch(batch_size)

  return train_dataset, val_dataset, test_dataset

class DataGenerator:
  def __init__(self, directory):
    self.directory=directory
  def __iter__(self):
    while True:
      files=find_files(self.directory)
      random.shuffle(files)
      for file in files:
        for chunk in pd.read_csv(file, header=None, sep=r",", chunksize=1000):
          X = tf.stack([tf.convert_to_tensor(chunk.iloc[:, 0].values * chunk.iloc[:, 1].values, dtype=tf.float32),
                        tf.convert_to_tensor(chunk.iloc[:, 0].values * chunk.iloc[:, 2].values, dtype=tf.float32),
                        tf.math.log(tf.convert_to_tensor(chunk.iloc[:, 3].values+1, dtype=tf.float32)) / tf.math.log(tf.constant(10.0))
          ])
          X = tf.transpose(X)
          X = tf.expand_dims(X, axis=1)
          y_abs = []
          y_p = []
          y_qty = []
          for i in range(3, chunk.shape[1], 3):
            y_abs.append(tf.math.log(tf.convert_to_tensor(chunk.iloc[:, 3].values+1, dtype=tf.float32)) / tf.math.log(tf.constant(10.0)))
            y_p.append(tf.convert_to_tensor(chunk.iloc[:, i + 1].values + np.pi, dtype=tf.float32))
            y_qty.append(tf.convert_to_tensor(chunk.iloc[:, i + 2].values, dtype=tf.float32))
          y_abs_tensor = tf.stack(y_abs, axis=1)
          y_p_tensor = tf.stack(y_p, axis=1)
          y_qty_tensor = tf.stack(y_qty, axis=1)
          y_a = tf.stack([y_abs_tensor, y_qty_tensor], axis=-1)
          y_phase = tf.stack([y_p_tensor, y_qty_tensor], axis=-1)
          for x, a, p in zip(X, y_a, y_phase):
            yield (x, (a, p))
      return

train_dataset, val_dataset, test_dataset = create_tf_dataset()

#def mae_qty(y_true, y_pred):
#  qty=y_true[:,:,2]
#  mae=tf.reduce_sum(tf.abs((y_true[:,:, :2]-y_pred)), axis=-1) 
#  loss=mae*qty
#  return tf.reduce_sum(loss)

def abs_qty(y_true, y_pred):
  qty=y_true[:, :,1]
  mae=tf.abs(y_true[:,:, 0]-y_pred[:, 0])
  loss=mae*qty/tf.reduce_max(y_true[:, :, 0])
  return tf.reduce_sum(loss)

def phase_qty(y_true, y_pred):
  qty=y_true[:,:,1]
  mae=tf.abs(y_true[:,:, 0]-y_pred[:, 1])
  loss=mae*qty/2/np.pi
  return tf.reduce_sum(loss)

#model = keras.Sequential()
input_layer=layers.Input(shape=(1,3))
lstm=layers.LSTM(128)(input_layer)
repeat=layers.RepeatVector(100)(lstm)
lstm2=layers.LSTM(128, return_sequences=True)(repeat)
abs_output=layers.TimeDistributed(layers.Dense(1, activation='linear'), name='abs_output')(lstm2)
phase_output=layers.TimeDistributed(layers.Dense(1, activation='linear'), name='phase_output')(lstm2)
#model.add(layers.LSTM(100, activation='tanh'))
#model.add(layers.RepeatVector(100))
#model.add(layers.LSTM(100, activation='tanh', return_sequences=True))
#model.add(layers.Dense(2, activation='linear'))
model = tf.keras.models.Model(inputs=input_layer, outputs=[abs_output, phase_output])
model.summary()

es=tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-2), loss={'abs_output': abs_qty, 'phase_output': phase_qty}, metrics=[abs_qty, phase_qty])

try:
    model.fit(train_dataset, validation_data=val_dataset, epochs=1000, callbacks=[es])
except KeyboardInterrupt:
    model.save("unfinished.keras")
model.save('prototype2.keras')

model.evaluate(test_dataset)
