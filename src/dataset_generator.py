from enum import Enum, auto
import numpy as np
import pandas as pd
import tensorflow as tf


class DatasetType(Enum):
    SIMPLE=auto()
    ENCODER=auto()
    DECODER_POINT=auto()
    DECODER_SEQUENCE=auto()

def filter_dataset(x, y):
  """
  Фильтрует набор данных, проверяя, что все значения в входном тензоре
  ``x`` и в тензорах ``y`` являются конечными (не NaN или inf).
  Parameters
  ----------
  x : Tensor
      Входной тензор, содержащий входные данные.
  y : Tensor/tuple of Tensor
      Тензор или кортеж из двух тензоров с размерностью (batch_size, 100, 2), представляющий собой выходные данные.
  Returns
  -------
  Bool
      True, если значения в тензорах конечные, иначе False.
  """
  if isinstance(y, tuple):
    return tf.reduce_all(tf.math.is_finite(x)) & tf.reduce_all(tf.math.is_finite(y[0])) & tf.reduce_all(tf.math.is_finite(y[1]))
  else:
    return tf.reduce_all(tf.math.is_finite(x)) & tf.reduce_all(tf.math.is_finite(y))

def create_tf_dataset(chunk, dataset_type, line_num=-1):
    data_gen = DataGenerator(chunk, dataset_type, line_num)
    match dataset_type:
        case dataset_type.SIMPLE:
            dataset = tf.data.Dataset.from_generator(
            lambda: data_gen,
            output_signature=(
                tf.TensorSpec(shape=(100, 2), dtype=tf.float32),
                (tf.TensorSpec(shape=(100, 2), dtype=tf.float32),
                 tf.TensorSpec(shape=(100, 2), dtype=tf.float32)))
            )
        case dataset_type.ENCODER:
            dataset = tf.data.Dataset.from_generator(
            lambda: data_gen,
            output_signature=(
                tf.TensorSpec(shape=(100, 2), dtype=tf.float32),
                tf.TensorSpec(shape=(1,3), dtype=tf.float32))
            )
        case dataset_type.DECODER_POINT:
            dataset = tf.data.Dataset.from_generator(
            lambda: data_gen,
            output_signature=(
                tf.TensorSpec(shape=(1, 4), dtype=tf.float32),
               (tf.TensorSpec(shape=(1, 2), dtype=tf.float32),
                tf.TensorSpec(shape=(1, 2), dtype=tf.float32)))
            )
            dataset=dataset.shuffle(buffer_size=99000)
        case dataset_type.DECODER_SEQUENCE:
            dataset = tf.data.Dataset.from_generator(
            lambda: data_gen,
            output_signature=(
                tf.TensorSpec(shape=(None, 4), dtype=tf.float32),
               (tf.TensorSpec(shape=(1, 2), dtype=tf.float32),
                tf.TensorSpec(shape=(1, 2), dtype=tf.float32)))
            )
        case _:
            raise AssertionError("Неизвестный тип датасета.")
    dataset=dataset.filter(filter_dataset)
    return dataset

class DataGenerator:
    def __init__(self, chunk, dataset_type, line_num):
        self.chunk = chunk
        self.dataset_type = dataset_type
        self.line_num = line_num
    def __iter__(self):
        match self.dataset_type:
            case self.dataset_type.SIMPLE:
                abs = []
                phases = []
                qty = []
                for i in range(3, self.chunk.shape[1], 3):
                    abs.append(tf.convert_to_tensor(np.log(self.chunk.iloc[:, i].values+1)/10, dtype=tf.float32))
                    phases.append(tf.convert_to_tensor((self.chunk.iloc[:, i + 1].values+np.pi)/2/np.pi, dtype=tf.float32))
                    qty.append(tf.convert_to_tensor(self.chunk.iloc[:, i + 2].values, dtype=tf.float32))
                abs_tensor = tf.stack(abs, axis=1)
                phases_tensor = tf.stack(phases, axis=1)
                qty_tensor = tf.stack(qty, axis=1)
                X = tf.stack([abs_tensor, phases_tensor], axis=-1)
                Y_abs = tf.stack([abs_tensor, qty_tensor], axis=-1)
                Y_phases = tf.stack([phases_tensor, qty_tensor], axis=-1)
                for x, abs, phase in zip(X, Y_abs, Y_phases):
                    yield (x, (abs, phase))
            case self.dataset_type.ENCODER:
                Y = tf.stack([tf.convert_to_tensor(np.log(self.chunk.iloc[:, 0].values * (self.chunk.iloc[:, 1].values + 10000))/25, dtype=tf.float32),
                              tf.convert_to_tensor(np.log(self.chunk.iloc[:, 0].values * (self.chunk.iloc[:, 2].values + 10000))/25, dtype=tf.float32),
                              tf.convert_to_tensor(np.log(self.chunk.iloc[:, 3].values+1)/10, dtype=tf.float32)
                ])
                Y = tf.transpose(Y)
                Y = tf.expand_dims(Y, axis=1)
                abs = []
                phases = []
                for i in range(3, self.chunk.shape[1], 3):
                    abs.append(tf.convert_to_tensor(np.log(self.chunk.iloc[:, i].values+1)/10, dtype=tf.float32))
                    phases.append(tf.convert_to_tensor((self.chunk.iloc[:, i + 1].values+np.pi)/2/np.pi, dtype=tf.float32))
                abs_tensor = tf.stack(abs, axis=1)
                phases_tensor = tf.stack(phases, axis=1)
                X = tf.stack([abs_tensor, phases_tensor], axis=-1)
                for x, y in zip(X, Y):
                  yield (x, y)
            case self.dataset_type.DECODER_POINT:
                X=[]
                Y_abs=[]
                Y_phase=[]
                for j in range(self.chunk.shape[0]):
                    for i in range(3, self.chunk.shape[1]-3, 3):
                        x=tf.reshape(tf.stack([
                            tf.convert_to_tensor(np.log(self.chunk.iloc[j, 0] * (self.chunk.iloc[j, 1] + 10000)) / 25, dtype=tf.float32),
                            tf.convert_to_tensor(np.log(self.chunk.iloc[j, 0] * (self.chunk.iloc[j, 2] + 10000)) / 25, dtype=tf.float32),
                            tf.convert_to_tensor(np.log(self.chunk.iloc[j, i] + 1) / 10, dtype=tf.float32),
                            tf.convert_to_tensor((self.chunk.iloc[j, i+1] + np.pi) / (2*np.pi), dtype=tf.float32)]), (1,4))
                        y_abs=tf.stack([
                            tf.convert_to_tensor(np.log(self.chunk.iloc[j, i+3] + 1) / 10, dtype=tf.float32),
                            tf.convert_to_tensor(self.chunk.iloc[j, i+5], dtype=tf.float32)
                        ])
                        y_phase = tf.stack([
                            tf.convert_to_tensor((self.chunk.iloc[j, i+4] + np.pi) / 2 / np.pi, dtype=tf.float32),
                            tf.convert_to_tensor(self.chunk.iloc[j, i+5], dtype=tf.float32)
                        ])
                        X.append(x)
                        Y_abs.append(tf.expand_dims(y_abs, axis=0))
                        Y_phase.append(tf.expand_dims(y_phase, axis=0))
                for x, y_abs, y_phase in zip(X, Y_abs, Y_phase):
                    yield (x, (y_abs, y_phase))
            case self.dataset_type.DECODER_SEQUENCE:
                X=[]
                Y_abs=[]
                Y_phase=[]
                for j in range(self.chunk.shape[0]):
                    x1=tf.convert_to_tensor(np.log(self.chunk.iloc[j, 0] * (self.chunk.iloc[j, 1] + 10000)) / 25, dtype=tf.float32),
                    x2=tf.convert_to_tensor(np.log(self.chunk.iloc[j, 0] * (self.chunk.iloc[j, 2] + 10000)) / 25, dtype=tf.float32),
                    x3=tf.keras.ops.reshape(tf.convert_to_tensor(np.log(self.chunk.iloc[j, 4:3*self.line_num+4:3] + 1) / 10, dtype=tf.float32), (-1,1)),
                    x4=tf.keras.ops.reshape(tf.convert_to_tensor((self.chunk.iloc[j, 5:3*self.line_num+5:3] + np.pi) / (2 * np.pi), dtype=tf.float32), (-1,1))
                    repeat_count = tf.shape(x3)[0]
                    x1=tf.expand_dims(x1, axis=0)
                    x2=tf.expand_dims(x2, axis=0)
                    x1_repeated = tf.keras.ops.repeat(x1, repeat_count, axis=0)
                    x2_repeated = tf.keras.ops.repeat(x2, repeat_count, axis=0)
                    x = tf.concat([x1_repeated, x2_repeated, x3, x4], axis=1)
                    X.append(x)
                    y_abs = tf.stack([tf.convert_to_tensor(np.log(self.chunk.iloc[j, 3*self.line_num + 6] + 1) / 10, dtype=tf.float32),
                                      tf.convert_to_tensor(chunk.iloc[j, 3*self.line_num + 8], dtype=tf.float32)
                    ])
                    y_phase = tf.stack([tf.convert_to_tensor((self.chunk.iloc[j, 3*self.line_num + 7] + np.pi) / (2 * np.pi), dtype=tf.float32),
                                        tf.convert_to_tensor(self.chunk.iloc[j, 3*self.line_num + 8], dtype=tf.float32)
                    ])
                    Y_abs.append(tf.expand_dims(y_abs, axis=0))
                    Y_phase.append(tf.expand_dims(y_phase, axis=0))
                for x, y_abs, y_phase in zip(X, Y_abs, Y_phase):
                    yield (x, (y_abs, y_phase))
            case _:
                raise AssertionError("Неизвестный тип датасета.")
