## Installation
We recommend using a virtual environment (Anaconda, virtualenv) to to control
package visibility. For a detailed list of requirements, see
[environment.yml](https://github.com/ad12/MedSegPy/blob/master/environment.yml).


### Requirements
- Python ≥3.6
- tensorflow(-gpu) ≥1.8.0
- keras ≥2.1.6
- pydot (download via `conda install pydot`)
- graphviz (download via `conda install python-graphviz`)

MedSegPy supports both TensorFlow 1 and 2. For gpu
tensorflow versions, compatible cuda drivers and toolkits
must be installed. If you do not have a gpu or you are using
`tensorflow>=2.0`, replace the `tensorflow-gpu` package with `tensorflow`.
More instructions about TensorFlow cuda compatibility and installation
can be found [here](https://www.tensorflow.org/install/source#gpu).

Install these requirements prior to installing MedSegPy.


### Build MedSegPy from Source
```bash
python -m pip install 'git+https://github.com/ad12/MedSegPy.git'
# (add --user if you don't have permission)

# Or, to install it from a local clone (recommended):
git clone https://github.com/ad12/MedSegPy.git
cd MedSegPy && python -m pip install -e .
```

You may need to rebuild medsegpy after reinstalling TensorFlow and/or Keras.


### Step-by-Step Guide
The following instructions install MedSegPy with Python 3.7,
Tensorflow 2.3, and Keras 2.4 into an Anaconda environment.
Note gpu use with Tensorflow 2.3 requires cuda toolkit 10.1.

```bash
# Create conda environment
conda create -n medsegpy_env python=3.7
conda activate medsegpy_env

# Install tensorflow and keras dependencies.
pip install tensorflow==2.3.1 keras==2.4.3

# Install visualization dependencies
conda install pydot python-graphviz

# Install MedSegPy
git clone https://github.com/ad12/MedSegPy.git
cd MedSegPy && python -m pip install -e .
```


### Configuring Paths (optional)
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

```python
from dosma.utils.cluster import Cluster

# Define paths for machineA
machineA = Cluster("machineA", ["machineA_hostname_pattern"], "path/to/machineA/datasets", "path/to/machineA/results")
machineA.save()

# Define paths for machineB
machineB = Cluster("machineB", ["machineB_hostname_pattern"], "path/to/machineB/datasets", "path/to/machineB/results")
machineB.save()
```
