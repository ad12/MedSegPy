## Getting Started with MedSegPy

This document provides a brief intro of the usage of builtin command-line tools in medsegpy.

For more advanced tutorials, refer to our [documentation](https://ad12.github.io/MedSegPy/).

### Training & Evaluation in Command Line

We provide a script in "medsegpy/nn_train.py", that is made to train
all the configs provided in medsegpy.
You may want to use it as a reference to write your own training script for a new research.

To train a model with "nn_train.py", first
setup the corresponding datasets following
[datasets/README.md](https://github.com/ad12/MedSegPy/tree/master/datasets),
then run:
```
python tools/train_net.py --num-gpus 1 \
	--config-file configs/OAI-iMorphics/unet_2d.ini \
	OUTPUT_DIR /PATH/TO/SAVE/DIR
```

For CPU training, use `--num-gpus 0`, though this is not recommended.

To evaluate a model's performance, use
```
python tools/train_net.py \
	--config-file /PATH/TO/SAVE/DIR/config.ini \
	--eval-only TEST_WEIGHT_PATH /path/to/checkpoint_file
```
For more options, see `python tools/train_net.py -h`.
