from keras import initializers


def get_initializer(initializer_name, seed=None):
    return initializers.get({'class_name': initializer_name, 'config': {'seed': seed}})
