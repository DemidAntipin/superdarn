import tensorflow as tf
from tensorflow.keras import layers
from src.serializable_functions import abs_qty, phase_qty
from src.dataset_generator import create_tf_dataset, DatasetType
from glob import glob
import os
import pandas as pd

input_layer=layers.Input(shape=(100, 2), name="input")
lstm=layers.LSTM(64, return_sequences=True, name="lstm_encoder_1")(input_layer)
lstm=layers.LSTM(16, name="lstm_encoder_2")(lstm)
encoder_output=layers.Dense(3, activation="sigmoid")(lstm)
repeat=layers.RepeatVector(100)(encoder_output)
lstm=layers.LSTM(128, return_sequences=True, name="lstm_decoder")(repeat)
abs_output=layers.TimeDistributed(layers.Dense(1, activation="sigmoid"), name="abs_output")(lstm)
phase_output=layers.TimeDistributed(layers.Dense(1, activation="sigmoid"), name="phase_output")(lstm)

autoencoder=tf.keras.models.Model(inputs=input_layer, outputs=[abs_output, phase_output])
autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss=[abs_qty,phase_qty], metrics=[abs_qty, phase_qty])

es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

def find_files(directory):
  return glob(os.path.join(directory, '*.bz2'))

directory='data/train/'

try:
    files=find_files(directory)
    for file in files:
        for chunk in pd.read_csv(file, compression='bz2', header=None, sep=r" ", chunksize=1000):
            chunk=chunk.iloc[:, :-1] # в данных лишний пробел в конце каждой строки
            data=create_tf_dataset(chunk, DatasetType.SIMPLE)
            train_data, val_data = tf.keras.utils.split_dataset(data, left_size=0.8)
            train_data = train_data.batch(32)
            val_data = val_data.batch(32)
            autoencoder.fit(train_data, validation_data=val_data, epochs=30, callbacks=[es])
except Exception as e:
    print(e)
autoencoder.save("Models/autoencoder_simple.keras")
