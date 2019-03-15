from resnet import ResNet50
from keras.models import Model
from models.refinenet.refine_module import refine_module
from keras.utils import plot_model
from keras.layers import Conv2D


RESNET_INFO = {
                'resnet50':{'layers': [0, 3, 36, 78, 140, -2],
                            'num_filters': [32, 64, 256, 512, 1024, 2048]}
              }


def refinenet_model(input_shape=None, backbone='resnet50'):
    info = RESNET_INFO[backbone]
    downsampling_layers = info['layers']
    num_filters = info['num_filters']

    m = ResNet50(include_top=False, weights=None, input_shape=input_shape)
    # chop off final ave pooling layer
    m = Model(inputs=m.inputs, outputs=m.layers[downsampling_layers[-1]-1].output)

    # Add refinenet modules
    assert len(downsampling_layers) == len(num_filters)

    curr_refine = None

    for i in range(len(downsampling_layers))[::-1]:
        x = m.layers[downsampling_layers[i]].output

        x_ins = [x]
        if curr_refine is not None:
            x_ins.append(curr_refine)

        if i == 0:
            num_filters_out = num_filters[i]
        else:
            num_filters_out = num_filters[i-1]

        curr_refine = refine_module(x_ins,
                                    num_filters_in = num_filters[i],
                                    num_filters_out=num_filters_out,
                                    name_prefix='dec_%d' % i)

    return Model(inputs=m.inputs, outputs=curr_refine)

if __name__ == '__main__':
    save_path = '../imgs/refinenet_resnet50.png'
    m = refinenet_model(input_shape=(288,288,1))
    plot_model(m, to_file=save_path)





