import numpy as np
from keras.initializers import glorot_uniform, he_normal
from keras.layers import Activation
from keras.layers import BatchNormalization as BN
from keras.layers import (
    Conv3D,
    Deconvolution3D,
    Dropout,
    Input,
    MaxPooling3D,
    concatenate,
)
from keras.models import Model
from keras.utils import plot_model

DEFAULT_INPUT_SIZE = (288, 288, 64, 1)


def unet_3d_model(
    input_size=DEFAULT_INPUT_SIZE,
    input_tensor=None,
    num_classes=1,
    activation="sigmoid",
    output_mode=None,
    num_filters=None,
    depth=6,
    filter_size=(3, 3, 3),
    in_plane_pool_size=(2, 2),
    seed=None,
):
    # input size is a tuple of the size of the image
    # assuming channel last
    # input_size = (dim1, dim2, dim3, ch)
    # unet begins
    if input_tensor is None and (
        type(input_size) is not tuple or len(input_size) != 4
    ):
        raise ValueError(
            "input_size must be a tuple of size (height, width, slices, 1)"
        )

    if num_filters is None:
        nfeatures = [2 ** feat * 32 for feat in np.arange(depth)]
    else:
        nfeatures = num_filters
        assert len(nfeatures) == depth

    xres, yres, slices, chans = input_size

    conv_ptr = []
    dim3 = []

    # Calculate what depth pooling should stop in dim3
    for _ in range(depth):
        slices /= 2
        dim3.append(int(np.floor(slices)))

    # input layer
    inputs = Input(input_size)

    # step down convolutional layers
    pool = inputs
    for depth_cnt in range(depth):

        conv = Conv3D(
            nfeatures[depth_cnt],
            filter_size,
            padding="same",
            kernel_initializer=he_normal(seed=seed),
        )(pool)
        conv = Activation("relu")(conv)
        conv = Conv3D(
            nfeatures[depth_cnt],
            filter_size,
            padding="same",
            kernel_initializer=he_normal(seed=seed),
        )(conv)
        conv = Activation("relu")(conv)

        conv = BN(axis=-1, momentum=0.95, epsilon=0.001)(conv)
        conv = Dropout(rate=0.0)(conv)

        conv_ptr.append(conv)

        # Pool in slice dimension only if dim has length => 2
        if dim3[depth_cnt] > 0:
            pool = MaxPooling3D(pool_size=in_plane_pool_size + (2,))(conv)
        else:
            pool = MaxPooling3D(pool_size=in_plane_pool_size + (1,))(conv)

    # step up convolutional layers
    for depth_cnt in range(depth - 2, -1, -1):

        deconv_shape = conv_ptr[depth_cnt].shape.as_list()
        deconv_shape[0] = None

        # Deconv in slice dimension only if slice dim has size > 2
        if dim3[depth_cnt] > 0:
            up = concatenate(
                [
                    Deconvolution3D(
                        nfeatures[depth_cnt],
                        filter_size,
                        padding="same",
                        strides=in_plane_pool_size + (2,),
                        # output_shape=deconv_shape,
                        kernel_initializer=he_normal(seed=seed),
                    )(conv),
                    conv_ptr[depth_cnt],
                ],
                axis=4,
            )

        else:
            up = concatenate(
                [
                    Deconvolution3D(
                        nfeatures[depth_cnt],
                        filter_size,
                        padding="same",
                        strides=in_plane_pool_size + (1,),
                        # output_shape=deconv_shape,
                        kernel_initializer=he_normal(seed=seed),
                    )(conv),
                    conv_ptr[depth_cnt],
                ],
                axis=4,
            )

        conv = Conv3D(
            nfeatures[depth_cnt],
            filter_size,
            padding="same",
            kernel_initializer=he_normal(seed=seed),
        )(up)
        conv = Activation("relu")(conv)
        conv = Conv3D(
            nfeatures[depth_cnt],
            filter_size,
            padding="same",
            kernel_initializer=he_normal(seed=seed),
        )(conv)
        conv = Activation("relu")(conv)

        conv = BN(axis=-1, momentum=0.95, epsilon=0.001)(conv)
        conv = Dropout(rate=0.00)(conv)

    # combine features
    recon = Conv3D(
        num_classes,
        (1, 1, 1),
        padding="valid",
        kernel_initializer=glorot_uniform(seed=seed),
    )(conv)
    recon = Activation(activation)(recon)

    model = Model(inputs=[inputs], outputs=[recon])

    return model


if __name__ == "__main__":
    m = unet_3d_model(input_size=(288, 288, 16, 1))
    plot_model(m, "./imgs/unet_3d.png", show_shapes=True)
