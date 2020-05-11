import logging

from keras import callbacks as kc
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import io
from PIL import Image

__all__ = ["lr_callback", "LossHistory", "GetAttnCoeff"]

logger = logging.getLogger(__name__)


def lr_callback(optimizer):
    """Wrapper for learning rate tensorflow metric.

    Args:
        optimizer: Optimizer used for training.

    Returns:
        func: To be wrapped in metric or callback.
    """

    def lr(y_true, y_pred):
        return optimizer.lr

    return lr


class LossHistory(kc.Callback):
    """A Keras callback to log training history
    """

    def on_train_begin(self, logs=None):
        self.val_losses = []
        self.losses = []
        self.epoch = []

    def on_epoch_end(self, epoch, logs=None):
        self.val_losses.append(logs.get("val_loss", float("nan")))
        self.losses.append(logs.get("loss"))
        self.epoch.append(epoch + 1)

        metrics = " - ".join(
            [
                "{}: {:0.4f}".format(k, v)
                if v >= 1e-3
                else "{}: {:0.4e}".format(k, v)
                for k, v in logs.items()
            ]
        )
        logger.info("Epoch {} - {}".format(epoch + 1, metrics))


class GetAttnCoeff(kc.Callback):

    def tf_summary_image(self, tensor):
        tensor = tensor.astype(np.uint8)
        height, width = tensor.shape
        channel = 1
        image = Image.fromarray(tensor)
        output = io.BytesIO()
        image.save(output, format='PNG')
        image_string = output.getvalue()
        output.close()
        return tf.Summary.Image(height=height,
                                width=width,
                                colorspace=channel,
                                encoded_image_string=image_string)

    def on_epoch_end(self, epoch, logs=None):
        _, attn_coeffs = self.model.get_layer('multi_attention_module2d_1').output
        coeffs_1 = attn_coeffs[-1, ..., 0]
        coeffs_1 = coeffs_1 * 255
        image = self.tf_summary_image(coeffs_1)
        summary = tf.Summary(value=[tf.Summary.Value(image=image)])
        writer = tf.summary.FileWriter('./logs')
        writer.add_summary(summary, epoch)
        writer.close()
        return

