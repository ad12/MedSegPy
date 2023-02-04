# Improving Data-Efficiency and Robustness of Medical Imaging Segmentation Using Inpainting-Based Self-Supervised Learning

**Paper:** Improving Data-Efficiency and Robustness of Medical Imaging Segmentation Using Inpainting-Based Self-Supervised Learning

**Authors:** Jeffrey Dominic, Nandita Bhaskar, Arjun D. Desai, Andrew Schmidt, Elka Rubin, Beliz Gunel, Garry E. Gold, Brian A. Hargreaves, Leon Lenchik, Robert Boutin, Akshay S. Chaudhari

This project includes the code for this paper.

## Usage
Example config files for pretraining and segmentation are included in the `configs` folder for the CT and MRI dataset.
The config files for the CT dataset are located in `configs/abct` and the config files 
for the MRI dataset are located in `configs/mri`.

Training and evaluations can be done using [tools/train_net.py](../../tools/train_net.py).

### Datasets
The MRI dataset is publicly available, 
and can be found [here](https://github.com/StanfordMIMI/skm-tea).

Please contact the authors for access to the CT dataset.

After obtaining the MRI and CT datasets, please make sure to change 
the paths of the datasets in [medsegpy/data/datasets/qdess_mri.py](../../medsegpy/data/datasets/qdess_mri.py) 
and [medsegpy/data/datasets/abct.py](../../medsegpy/data/datasets/abct.py).

### SSL Pretraining
A model for inpainting can be trained for either the CT or MRI dataset using the appropriate
pretraining config file.

For example, to train a model for inpainting for the MRI dataset, 
using 32 x 32 patches, Poisson-disc sampling, and the context restoration pretext task, 
you can run:

```bash
python /path/to/MedSegPy/tools/train_net.py --num-gpus 1 --config-file configs/mri/swap_patches/pretrain_inpainting/32_size.yaml
```

### SSL Segmentation
After pretraining, a model for segmentation can be trained using the appropriate 
config file.

For example, to train a model to segment tissues for the MRI dataset using 10% of the available
training data after the above pretraining, you can run:

```bash
python /path/to/MedSegPy/tools/train_net.py --num-gpus 1 --config-file configs/mri/swap_patches/segmentation/32_size.yaml
```

Please make sure to set the parameter `PRETRAINED_CONFIG_PATH` in the config file 
to the path of the config file for the pretrained model.

### Fully-Supervised Segmentation
A baseline fully-supervised model can be trained using the appropriate config file.

For example, to train a fully-supervised model to segment tissues for the MRI dataset using 
10% of the available training data, you can run:

```bash
python /path/to/MedSegPy/tools/train_net.py --num-gpus 1 --config-file configs/mri/fully_supervised/10_perc.yaml
```