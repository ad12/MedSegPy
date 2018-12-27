# Base unet - best performing network
BASE_UNET = '/bmrNAS/people/arjun/msk_seg_networks/architecture_limit/unet_2d/original_akshaysc/test_results'

# Non-fine-tuned unet - unet without any fine-tuning
NON_FT_UNET = '/bmrNAS/people/arjun/msk_seg_networks/architecture_limit/unet_2d/2018-11-26-00-56-55/test_results'

# non-fine-tuned unet - no -augmentation
BASELINE_UNET = '/bmrNAS/people/arjun/msk_seg_networks/augment_limited/unet_2d/2018-11-07-16-05-15/test_results'

# Arch No finetune
ARCH_UNET_RAW = BASELINE_UNET
ARCH_SEGNET_RAW = '/bmrNAS/people/arjun/msk_seg_networks/architecture_limit/segnet_2d/2018-11-30-21-13-14/test_results'
ARCH_DEEPLAB_RAW = '/bmrNAS/people/arjun/msk_seg_networks/architecture_limit/deeplabv3_2d/2018-11-30-05-49-49/test_results/16_2-4-6'

# Architecture experiments
ARCH_UNET = '/bmrNAS/people/arjun/msk_seg_networks/augment_limited/unet_2d/2018-11-07-16-05-15/fine_tune/test_results'
ARCH_SEGNET = '/bmrNAS/people/arjun/msk_seg_networks/architecture_limit/segnet_2d/2018-11-30-21-13-14/fine_tune/test_results'
ARCH_DEEPLAB = '/bmrNAS/people/arjun/msk_seg_networks/architecture_limit/deeplabv3_2d/2018-11-30-05-49-49/fine_tune/test_results/16_2-4-6'

# loss functions experiments
LOSS_DSC = BASELINE_UNET
LOSS_WCE = '/bmrNAS/people/arjun/msk_seg_networks/loss_limit/unet_2d/2018-12-11-21-59-16/test_results/'
LOSS_BCE = '/bmrNAS/people/arjun/msk_seg_networks/loss_limit/unet_2d/2018-11-28-08-07-03/test_results'
LOSS_FOCAL = '/bmrNAS/people/arjun/msk_seg_networks/loss_limit/unet_2d/2018-11-28-19-33-22/test_results'

# augmentation experiments
AUG_YES = '/bmrNAS/people/arjun/msk_seg_networks/architecture_limit/unet_2d/2018-11-26-00-56-55/test_results'
AUG_NO = BASELINE_UNET

# volumetric architecture experiments
VOLUME_2D = AUG_NO
VOLUME_2_5D_3 = '/bmrNAS/people/arjun/msk_seg_networks/volume_limited/unet_2_5d/2018-10-26-20-57-21/test_results' # 2.5D network with 3 slices (+/-1)
VOLUME_2_5D_5 = '/bmrNAS/people/arjun/msk_seg_networks/volume_limited/unet_2_5d/2018-10-21-02-31-40/test_results' # 2.5D network with 5 slices (+/-2)
VOLUME_2_5D_7 = '/bmrNAS/people/arjun/msk_seg_networks/volume_limited/unet_2_5d/2018-10-26-21-00-40/test_results' # 2.5D network with 7 slices (+/-3)
VOLUME_3D = '/bmrNAS/people/arjun/msk_seg_networks/volume_limited/unet_3d_asc/test_results' # 3D network with 4 slice inputs