import numpy as np
import matplotlib.pyplot as pp
import tensorflow as tf
from src.serializable_functions import abs_qty, phase_qty, extract_first_param
from src.dataset_generator import create_tf_dataset, DatasetType
from src.autoencoder_seq import Autoencoder_seq
from src.autoencoder_point import Autoencoder_point
from glob import glob
import os
import pandas as pd
import bz2

def find_files(directory):
  return glob(os.path.join(directory, '*.bz2'))

directory='data/test/'

files=find_files(directory)

encoder=tf.keras.models.load_model("Models/encoder.keras")
seq_decoder=tf.keras.models.load_model("Models/decoder_seq.keras")
seq_autoencoder=Autoencoder_seq(encoder, seq_decoder)
seq_autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss=[abs_qty, phase_qty], metrics=[abs_qty, phase_qty])
point_decoder=tf.keras.models.load_model("Models/decoder_point.keras")
point_autoencoder=Autoencoder_point(encoder, point_decoder)
point_autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss=[abs_qty, phase_qty], metrics=[abs_qty, phase_qty])

models=[point_autoencoder, seq_autoencoder]
print("Выберите модель:\n", "1: point_autoencoder\n", "2: seq_autoencoder")
index=int(input())
model=models[index-1]

try:
    for file in files:
        for chunk in pd.read_csv(file, compression='bz2', header=None, sep=r" ", chunksize=1000):
            chunk=chunk.iloc[:, :-1] # в данных лишний пробел в конце каждой строки
            test_data=create_tf_dataset(chunk, DatasetType.SIMPLE)
            test_data=test_data.batch(32)
            X = test_data.map(lambda x, y: x)
            results=model.evaluate(test_data)
            #names=model.metrics_names
            #for metric, value in zip(names, results):
            #    print(f"{metric}: {value:.5f}")
            for x in X:
                for i in range(x.shape[0]):
                    prediction=model.predict(x[i:i+1])
                    fig, axs = pp.subplots(1, 2, constrained_layout=True, figsize=(8, 4))
                    axs[0].set_title('Abs Prediction')
                    axs[0].plot(x[i, :, 0], label='Actual Abs', color='blue')
                    axs[0].plot(prediction[:, 0], color='green', label='Predicted Abs')
                    axs[0].legend()
                    axs[1].set_title('Phase Prediction')
                    axs[1].plot(x[i, :, 1], label='Actual Phase', color='blue')
                    axs[1].plot(prediction[:, 1], color='green', label='Predicted Phase')
                    axs[1].legend()
                    pp.show()
except Exception as e:
    print(e)
