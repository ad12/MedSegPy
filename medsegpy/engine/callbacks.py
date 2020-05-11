import logging

from keras import callbacks as kc
import matplotlib.pyplot as plt

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
    def on_epoch_end(self, epoch, logs=None):
        _, attn_coeffs = self.model.get_layer('multi_attention_module2d_1').output
        #input_data = self.model.get_layer('conv2d_1').input
        coeffs_1 = attn_coeffs[-1, ..., 0]
        #input_last = input_data[-1, ..., 0]
        #attn_hmap = plt.imshow(coeffs_1, cmap='jet', interpolation='nearest',
        #                       vmin=0, vmax=1)
        print("Saving Attention Coefficient...")
        plt.imsave('/home/paperspace/attn_coeff_imgs/coeff_%d' % epoch,
                   coeffs_1, cmap='jet', vmin=0, vmax=1)
        #input_last = plt.imshow()
