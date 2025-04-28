import tensorflow as tf
import numpy as np

class Autoencoder_point(tf.keras.models.Model):
    def __init__(self, encoder_model, decoder_model):
        super(Autoencoder_point, self).__init__()
        self.encoder = encoder_model
        self.decoder = decoder_model
    def call(self, inputs):
        params = self.encoder(inputs)
        points = [(params[0][0][2], 0.5)]
        for i in range(99):
            p_x = tf.convert_to_tensor([[params[0][0][0], params[0][0][1], points[-1][0], points[-1][1]]])
            p_x = tf.expand_dims(p_x, axis=1)
            next_point = self.decoder(p_x)
            points.append((tf.squeeze(next_point[0]), tf.squeeze(next_point[1])))
        points_array = tf.convert_to_tensor(points)
        return points_array
    def evaluate(self, dataset, **kwargs):
        total_abs_qty = 0.0
        total_phase_qty = 0.0
        total_count = 0.0
        for x, y in dataset:
          for i in range(x.shape[0]):
              preds = self.call(tf.convert_to_tensor(x[i:i+1, :, :]))
              total_abs_qty += self.abs_qty(tf.convert_to_tensor(y[0]), preds[:, 0])
              total_phase_qty += self.phase_qty(tf.convert_to_tensor(y[1]), preds[:, 1])
              total_count+=1.0
        return total_abs_qty/total_count, total_phase_qty/total_count
    def abs_qty(self, y_true, y_pred):
        qty=y_true[:, :,1]
        mae=tf.abs(y_true[:,:, 0]-y_pred)
        loss=mae*qty
        return tf.reduce_mean(loss)
    def phase_qty(self, y_true, y_pred):
        qty=y_true[:,:,1]
        mae=tf.abs(y_true[:,:, 0]-y_pred)
        loss=mae*qty
        return tf.reduce_mean(loss)
