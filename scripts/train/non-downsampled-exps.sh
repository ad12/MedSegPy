#!/bin/bash

# List of experiments to run with non-downsampled data (i.e. 160 slices)
#
# @usage (from terminal/command line):
# ./non-downsampled-exps GPU_ID(s)
# eg: "./non-downsampled-exps 0", "./non-downsampled-exps 0,1"
#
# @initialization protocol:
#   1. run "chmod +x non-downsampled-exps" from the command line
#
# @author: Arjun Desai, Stanford University
#          (c) Stanford University, 2019

TOTAL_EXPS=(1)

if [ $# -lt 1 ]; then
	echo "Please provide gpus to run inference on in format `0` for single gpu or `0,1,2` for multiple gpus"
	exit 125
fi
GPU=$1

# Navigate to project directory
cd ../..


NEW_PATHS_ARGS='--train_path /bmrNAS/people/akshay/dl/oai_data/oai_3d/train --valid_path /bmrNAS/people/akshay/dl/oai_data/oai_3d/valid --test_path /bmrNAS/people/akshay/dl/oai_data/oai_3d/test'



# --------------- 2D UNet training experiment (control) ----------------- #
TRAIN_ARGS='--n_epochs 100 --use_step_decay --initial_learning_rate 0.02 --experiment non-downsampled -g '$GPU
ARGS=$TRAIN_ARGS' '$NEW_PATHS_ARGS

# Experiment 1: 2D unet (num_filters == [32, 64, 128, 256, 512, 1024])
#python -m oai_train unet_2d $ARGS --num_filters '[32, 64, 128, 256, 512, 1024]' --train_batch_size 12
#python -m oai_train unet_2d $ARGS --num_filters '[32, 64, 128, 256, 512, 1024]' --train_batch_size 35

# Experiment 2: 2D unet (num_filters == [16, 32, 64, 128, 256, 512])
#python -m oai_train unet_2d $ARGS --num_filters '[16, 32, 64, 128, 256, 512]'


# --------------- 2D DeepLabV3+ training experiment----------------- #
python -m oai_train deeplabv3_2d $NEW_PATHS_ARGS --initial_learning_rate 0.0001 --n_epochs 100 -g $GPU --experiment non-downsampled

# --------------- 2D Segnet+ training experiment----------------- #
python -m oai_train segnet_2d $NEW_PATHS_ARGS --initial_learning_rate 0.001 --n_epochs 100 -g $GPU --experiment non-downsampled

# --------------- Loss experiments----------------- #
TRAIN_ARGS='--n_epochs 100 --use_step_decay --initial_learning_rate 0.02 --experiment non-downsampled/loss_funcs -g '$GPU
ARGS=$TRAIN_ARGS' '$NEW_PATHS_ARGS

python -m oai_train unet_2d $ARGS --loss BINARY_CROSS_ENTROPY_SIG_LOSS
python -m oai_train unet_2d $ARGS --loss FOCAL_LOSS
python -m oai_train unet_2d $ARGS --loss FOCAL_LOSS


exit

# --------------- 3D training experiments----------------- #
TRAIN_ARGS='--tag oai_3d --n_epochs 100 --use_step_decay --drop_factor 0.6 --drop_rate 4 --initial_learning_rate 0.001 --experiment non-downsampled --train_batch_size 1 -g '$GPU
ARGS=$TRAIN_ARGS' '$NEW_PATHS_ARGS

# Experiment 3: 3D unet (4 slices)
#python -m oai_train unet_3d --img_size '(288,288,4,1)' $ARGS

# Experiment 4: 3D unet (8 slices)
#python -m oai_train unet_3d --img_size '(288,288,8,1)' $ARGS

# Experiment 5: 3D unet (16 slices)
#python -m oai_train unet_3d --img_size '(288,288,16,1)' $ARGS

# Experiment 6: 3D unet (32 slices)
#python -m oai_train unet_3d --img_size '(288,288,32,1)' $ARGS



# --------------- 3D block training experiments ----------------- #
TRAIN_ARGS='--tag oai_3d_block --n_epochs 100 --use_step_decay --drop_factor 0.6 --drop_rate 6 --initial_learning_rate 0.001 --experiment non-downsampled --train_batch_size 12 -g '$GPU
ARGS=$TRAIN_ARGS" "$NEW_PATHS_ARGS
echo $ARGS
# Experiment 7: 3D unet (4 slices)
#python -m oai_train unet_3d --num_filters '[32, 64, 128, 256, 512, 1024]' --img_size '(96,96,4,1)' $ARGS

# Experiment 8: 3D unet (8 slices)
#python -m oai_train unet_3d --num_filters '[32, 64, 128, 256, 512, 1024]' --img_size '(96,96,8,1)' $ARGS

# Experiment 9: 3D unet (16 slices)
python -m oai_train unet_3d --num_filters '[32, 64, 128, 256, 512, 1024]' --img_size '(96,96,32,1)' $ARGS

# Experiment 10: 3D unet (32 slices)
#python -m oai_train unet_3d --num_filters '[32, 64, 128, 256, 512, 1024]' --img_size '(96,96,32,1)' $ARGS

# Experiment 10: 3D unet (4 slices)
#python -m oai_train unet_3d --num_filters '[32, 64, 128, 256, 512, 1024]' --img_size '(96,96,64,1)' $ARGS



# --------------- 3D block training experiments (downsampled) ----------------- #
#TRAIN_ARGS='--tag oai_3d_block --n_epochs 100 --use_step_decay --drop_factor 0.6 --drop_rate 4 --initial_learning_rate 0.001 --experiment block --train_batch_size 12 -g '$GPU
#ARGS=$TRAIN_ARGS

# Experiment 7: 3D unet (4 slices)
#python -m oai_train unet_3d --num_filters '[32, 64, 128, 256, 512, 1024]' --img_size '(96,96,4,1)' $ARGS

# Experiment 8: 3D unet (4 slices)
#python -m oai_train unet_3d --num_filters '[32, 64, 128, 256, 512, 1024]' --img_size '(96,96,8,1)' $ARGS

# Experiment 9: 3D unet (4 slices)
#python -m oai_train unet_3d --num_filters '[32, 64, 128, 256, 512, 1024]' --img_size '(96,96,16,1)' $ARGS

# Experiment 10: 3D unet (4 slices)
#python -m oai_train unet_3d --num_filters '[32, 64, 128, 256, 512, 1024]' --img_size '(96,96,32,1)' $ARGS


