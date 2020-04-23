from keras.utils import plot_model

from medsegpy.modeling.segnet_2d.segnet_bottleneck import SegNetBottleneck

# Plot SegNet with single bottleneck
model = SegNetBottleneck(
    input_shape=(384, 384, 1),
    n_labels=4,
    depth=6,
    num_conv_layers=[2, 2, 3, 3, 3, 3],
    num_filters=[64, 128, 256, 256, 512, 512],
    output_mode="sigmoid",
)
model = model.build_model()
plot_model(
    model, show_shapes=True, to_file="../model_imgs/segnet_bottleneck.png"
)
