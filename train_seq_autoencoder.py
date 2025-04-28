import tensorflow as tf
from tensorflow.keras import layers
from src.serializable_functions import abs_qty, phase_qty, extract_first_param
from src.dataset_generator import create_tf_dataset, DatasetType
from glob import glob
import os
import pandas as pd

point_input = layers.Input(shape=(None,4), name="point_input")
lstm_point = layers.LSTM(64, activation='tanh', name='lstm_point', return_sequences=True, return_state=True)
all_state_h, state_h, state_c = lstm_point(point_input)
attention = layers.MultiHeadAttention(num_heads=4, key_dim=64)(all_state_h, all_state_h)
all_state_h = layers.Add()([all_state_h, attention])
states=[state_h, state_c]
lstm_point = layers.LSTM(64, activation='tanh', name='lstm_point2', return_sequences=True, return_state=True)
all_state_h, state_h, state_c = lstm_point(all_state_h, initial_state=states)
attention = layers.MultiHeadAttention(num_heads=4, key_dim=64)(all_state_h, all_state_h)
all_state_h = layers.Add()([all_state_h, attention])
states=[state_h, state_c]
lstm_point = layers.LSTM(64, activation='tanh', name='lstm_point_2')(all_state_h, initial_state=states)
point_abs = layers.Dense(1, activation='sigmoid', name='point_abs')(lstm_point)
point_phase = layers.Dense(1, activation='sigmoid', name='point_phase')(lstm_point)
seq_decoder = tf.keras.models.Model(inputs=point_input, outputs={"point_abs":point_abs, "point_phase": point_phase}, name="Point_decoder")
seq_decoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss={"point_abs":abs_qty, "point_phase": phase_qty}, metrics={"point_abs":abs_qty, "point_phase": phase_qty})

es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

def find_files(directory):
  return glob(os.path.join(directory, '*.bz2'))

directory='data/train/'

try:
    files=find_files(directory)
    for file in files:
        for chunk in pd.read_csv(file, compression='bz2', header=None, sep=r" ", chunksize=1000):
            chunk=chunk.iloc[:, :-1] # в данных лишний пробел в конце каждой строки
            for i in range(99):
              data=create_tf_dataset(chunk, DatasetType.DECODER_SEQUENCE, i)
              train_data, val_data = tf.keras.utils.split_dataset(data, left_size=0.8)
              train_data = train_data.batch(32)
              val_data = val_data.batch(32)
              seq_decoder.fit(train_data, validation_data=val_data, epochs=30, callbacks=[es])
except Exception as e:
    print(e)
seq_decoder.save("Models/decoder_seq.keras")
