## Getting Started with MedSegPy

This document provides a brief intro of the usage of builtin command-line tools in medsegpy.

For more advanced tutorials, refer to our [documentation]().


### Training & Evaluation in Command Line

We provide a script in "medsegpy/nn_train.py", that is made to train
all the configs provided in medsegpy.
You may want to use it as a reference to write your own training script for a new research.

To train a model with "nn_train.py", first
setup the corresponding datasets following
[datasets/README.md](https://github.com/ad12/MedSegPy/tree/master/configs),
then run:
```
python medsegpy/train_net.py --num-gpus 8 \
	--config-file configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml
```

The configs are made for 8-GPU training. To train on 1 GPU, change the batch size with:
```
python tools/train_net.py \
	--config-file configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml \
	SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.0025
```

For most models, CPU training is not supported.

(Note that we applied the [linear learning rate scaling rule](https://arxiv.org/abs/1706.02677)
when changing the batch size.)

To evaluate a model's performance, use
```
python tools/train_net.py \
	--config-file configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml \
	--eval-only MODEL.WEIGHTS /path/to/checkpoint_file
```
For more options, see `python tools/train_net.py -h`.

### Use Detectron2 APIs in Your Code

See our [Colab Notebook](https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5)
to learn how to use detectron2 APIs to:
1. run inference with an existing model
2. train a builtin model on a custom dataset

See [detectron2/projects](https://github.com/facebookresearch/detectron2/tree/master/projects)
for more ways to build your project on detectron2.
