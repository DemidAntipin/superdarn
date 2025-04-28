import tensorflow as tf
from tensorflow.keras import layers
from src.serializable_functions import abs_qty, phase_qty
from src.dataset_generator import create_tf_dataset, DatasetType
from glob import glob
import os
import pandas as pd

input_layer = layers.Input(shape=(100, 2), name="Input_Signal")
abs_0 = input_layer[:, 0, 0]
abs_0 = tf.keras.layers.Reshape((1,1))(abs_0)
lstm_encoder = layers.LSTM(64, activation='tanh', return_sequences=True, return_state=True, dropout=0.2)
all_state_h, state_h, state_c = lstm_encoder(input_layer)
states=[state_h, state_c]
lstm_encoder = layers.LSTM(64, activation='tanh', return_sequences=True, return_state=True, dropout=0.2)
all_state_h, state_h, state_c = lstm_encoder(all_state_h, initial_state=states)
states=[state_h, state_c]
lstm_encoder = layers.LSTM(64, activation='tanh', dropout=0.2)(all_state_h, initial_state=states)
latent = layers.Dense(2, activation='sigmoid', name="Latent_Features")(lstm_encoder)
reshaped_latent = layers.Reshape((1,2))(latent)
reshaped_latent = layers.Concatenate(axis=2)([reshaped_latent, abs_0])
encoder = tf.keras.models.Model(inputs=input_layer, outputs=reshaped_latent, name="Encoder")
encoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss='mae', metrics=['mae'])
encoder.summary()

es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

def find_files(directory):
  return glob(os.path.join(directory, '*.bz2'))

directory='data/train/'

try:
    files=find_files(directory)
    for file in files:
        for chunk in pd.read_csv(file, compression='bz2', header=None, sep=r" ", chunksize=1000):
            chunk=chunk.iloc[:, :-1] # в данных лишний пробел в конце каждой строки
            data=create_tf_dataset(chunk, DatasetType.ENCODER)
            train_data, val_data = tf.keras.utils.split_dataset(data, left_size=0.8)
            train_data = train_data.batch(32)
            val_data = val_data.batch(32)
            encoder.fit(train_data, validation_data=val_data, epochs=30, callbacks=[es])
except Exception as e:
    print(e)
encoder.save("Models/encoder.keras")
