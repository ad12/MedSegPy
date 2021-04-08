## Getting Started with MedSegPy

This document provides a brief intro of the usage of builtin command-line tools in medsegpy.

For more advanced tutorials, refer to our [documentation](https://ad12.github.io/MedSegPy/_build/html/index.html).


### Minimal Setup
##### Managing Paths
There are two primary paths that MedSegPy uses: 1. dataset paths,
the path to the directory holding the datasets, and 2. result paths,
the path to the directory where results should be stored.

***Builtin Dataset Paths***

You can set the location for builtin datasets by
export MEDSEGPY_DATASETS=/path/to/datasets. If left unset, the default
is ./datasets relative to your current working directory.

***Result Paths***

Similarly, you can set the location for the results directory by
export MEDSEGPY_RESULTS=/path/to/results. If left unset, the default
is ./results relative to your current working directory.

As a shortcut, we designate the prefix `"results://"`
in any filepath to point to a result directory of your choosing.
For example, `"results://exp1"` will resolve to the path
`"<MEDSEGPY_RESULTS>/exp1"`.

An example of how to do this in python (i.e. without export statements) is shown below:

```python
import os
os.environ["MEDSEGPY_DATASETS"] = "/path/to/datasets"
os.environ["MEDSEGPY_RESULTS"] = "/path/to/results"

import medsegpy.utils  # import implicitly registers prefixes
from fvcore.common.file_io import PathManager
PathManager.get_local_path("results://exp1")  # returns "/path/to/results/exp1"
```

You can also define your own prefixes to resolve by adding your own path handler.
This is useful if you want to use the same script to run multiple projects. See fvcore's
[fileio](https://github.com/facebookresearch/fvcore/blob/master/fvcore/common/file_io.py)
for more information.

***Using Clusters***
It can often be difficult to manage multiple machines may be used.
For example, data in `machineA` could be stored at a different location
than on `machineB`, but the code you have written is visible to both machines.
In these scenarios, managing centralized code can become quite challenging
because of hardcoded paths.

We can use `Cluster` to store our preferences for paths on different machines.
The code below saves the path preferences for each machine to a file.
Whenever these machines are detected (matched using hostname) when using MedSegPy,
the machine-specific datasets and results paths will be use.

To register these machines, you will have to find the regex pattern(s) that can be used
to uniquely identify the machine or set of machines you want to add functionality for.
See [medsegpy/utils/cluster.py](medsegpy/utils/cluster.py) for more details.

```python
from dosma.utils.cluster import Cluster

# Define paths for machineA
machineA = Cluster("machineA", ["machineA_hostname_pattern"], "path/to/machineA/datasets", "path/to/machineA/results")
machineA.save()

# Define paths for machineB
machineB = Cluster("machineB", ["machineB_hostname_pattern"], "path/to/machineB/datasets", "path/to/machineB/results")
machineB.save()
```


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
