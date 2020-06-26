## Getting Started with MedSegPy

This document provides a brief intro of the usage of builtin command-line tools in medsegpy.

For more advanced tutorials, refer to our [documentation](https://ad12.github.io/MedSegPy/).


### Minimal Setup
##### Registering New Users
To register users to existing machines/clusters, add your username and machines to support with that username to the `_USER_PATHS` dictionary in
[medsegpy/utils/cluster.py](medsegpy/utils/cluster.py).

To register new machines, you will have to find the regex pattern(s) that can be used to uniquely identify the machine or set of machines you want to add functionality for. See [medsegpy/utils/cluster.py](medsegpy/utils/cluster.py) for more details.


### Training & Evaluation in Command Line

We provide a script in "medsegpy/train_net.py", that is made to train
all the configs provided in medsegpy.
You may want to use it as a reference to write your own training script for
new research.

To train a model with "train_net.py", first
setup the corresponding datasets following
[datasets/README.md](datasets/),
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
