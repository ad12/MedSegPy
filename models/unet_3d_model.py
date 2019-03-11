import numpy as np
from keras.initializers import he_normal
from keras.layers import BatchNormalization as BN
from keras.layers import Deconvolution3D
from keras.layers import Input, Conv3D, MaxPooling3D, concatenate, Dropout
from keras.models import Model

DEFAULT_INPUT_SIZE = (288, 288, 64)


def unet_3d_model(input_size=DEFAULT_INPUT_SIZE, input_tensor=None, output_mode=None, num_filters=None, depth=6,
                  filter_size=(3, 3, 3), in_plane_pool_size=(2, 2)):
    # input size is a tuple of the size of the image
    # assuming channel last
    # input_size = (dim1, dim2, dim3, ch)
    # unet begins
    import glob_constants
    print('Initializing unet with seed: %s' % str(glob_constants.SEED))
    SEED = glob_constants.SEED
    if input_tensor is None and (type(input_size) is not tuple or len(input_size) != 3):
        raise ValueError('input_size must be a tuple of size (height, width, 1)')

    if num_filters is None:
        nfeatures = [2 ** feat * 32 for feat in np.arange(depth)]
    else:
        nfeatures = num_filters
        assert len(nfeatures) == depth

    xres, yres, slices, chans = input_size

    conv_ptr = []
    dim3 = []

    # Calculate what depth pooling should stop in dim3
    for cnt in range(depth):
        slices /= 2
        dim3.append(int(np.floor(slices)))

    # input layer
    inputs = Input(input_size)

    # step down convolutional layers
    pool = inputs
    for depth_cnt in range(depth):

        conv = Conv3D(nfeatures[depth_cnt], filter_size,
                      padding='same',
                      activation='relu',
                      kernel_initializer=he_normal(seed=SEED))(pool)
        conv = Conv3D(nfeatures[depth_cnt], filter_size,
                      padding='same',
                      activation='relu',
                      kernel_initializer=he_normal(seed=SEED))(conv)

        conv = BN(axis=-1, momentum=0.95, epsilon=0.001)(conv)
        conv = Dropout(rate=0.0)(conv)

        conv_ptr.append(conv)

        # Pool in slice dimension only if dim has length => 2
        if (dim3[depth_cnt] > 0):
            pool = MaxPooling3D(pool_size=in_plane_pool_size + (2,))(conv)
        else:
            pool = MaxPooling3D(pool_size=in_plane_pool_size + (1,))(conv)

    # step up convolutional layers
    for depth_cnt in range(depth - 2, -1, -1):

        deconv_shape = conv_ptr[depth_cnt].shape.as_list()
        deconv_shape[0] = None

        # Deconv in slice dimension only if slice dim has size > 2
        if (dim3[depth_cnt] > 0):
            up = concatenate([Deconvolution3D(nfeatures[depth_cnt], filter_size,
                                              padding='same',
                                              strides=in_plane_pool_size + (2,),
                                              output_shape=deconv_shape,
                                              kernel_initializer=he_normal(seed=SEED))(conv),
                              conv_ptr[depth_cnt]],
                             axis=4)

        else:
            up = concatenate([Deconvolution3D(nfeatures[depth_cnt], filter_size,
                                              padding='same',
                                              strides=in_plane_pool_size + (1,),
                                              output_shape=deconv_shape,
                                              kernel_initializer=he_normal(seed=SEED))(conv),
                              conv_ptr[depth_cnt]],
                             axis=4)

        conv = Conv3D(nfeatures[depth_cnt], filter_size,
                      padding='same',
                      activation='relu',
                      kernel_initializer=he_normal(seed=SEED))(up)
        conv = Conv3D(nfeatures[depth_cnt], filter_size,
                      padding='same',
                      activation='relu',
                      kernel_initializer=he_normal(seed=SEED))(conv)

        conv = BN(axis=-1, momentum=0.95, epsilon=0.001)(conv)
        conv = Dropout(rate=0.00)(conv)

    # combine features
    recon = Conv3D(1, (1, 1, 1), padding='same', kernel_initializer=he_normal(seed=SEED))(conv)

    model = Model(inputs=[inputs], outputs=[recon])

    return model
