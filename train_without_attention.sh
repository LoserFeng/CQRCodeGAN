#!/bin/bash

# Set the environment variable
export CUDA_VISIBLE_DEVICES=1

# Run the Python script with the specified arguments in the background and save output to logs/train.log
nohup python train.py \
    --src_dset /local4/fyf/dataset/uw_enhance_RG/A \
    --tgt_dset /local4/fyf/dataset/uw_enhance_RG/B \
    --outputs_dir ./outputs \
    --vgg_pretrained_path checkpoints/vgg \
    --no_dropout \
    --name CQR_GAN_without_attention \
    --model single \
    --dataset_mode unaligned \
    --which_model_netG sid_unet_resize \
    --which_model_netD no_norm_4 \
    --patchD \
    --patch_vgg \
    --patchD_3 5 \
    --n_layers_D 5 \
    --n_layers_patchD 4 \
    --fineSize 768 \
    --patchSize 128 \
    --skip 1 \
    --batchSize 4 \
    --use_norm 1 \
    --use_wgan 0 \
    --use_ragan \
    --hybrid_loss \
    --times_residual \
    --instance_norm 0 \
    --vgg 1 \
    --vgg_w 1. \
    --vgg_choose relu5_1 \
    --gpu_ids 0 \
    --resize_or_crop resize_and_crop > logs/train_without_attention.log 2>&1 &