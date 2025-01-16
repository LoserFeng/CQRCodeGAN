
export CUDA_VISIBLE_DEVICES=1

#!/bin/bash

nohup  python train.py \
    --src_dset /local4/fyf/dataset/uw_enhance_RG/A \
    --tgt_dset /local4/fyf/dataset/uw_enhance_RG/B \
    --outputs_dir ./outputs \
    --vgg_pretrained_path checkpoints/vgg \
    --no_dropout \
    --name CQRCodeGAN \
    --model CQRCodeGAN \
    --dataset_mode unaligned \
    --which_model_netG resnet_9blocks \
    --which_model_netD no_norm_4 \
    --patch_D \
    --patch_D_3 4 \
    --n_layers_D 5 \
    --n_layers_patchD 4 \
    --lambda_identity 1 \
    --fineSize 768 \
    --patchSize 64 \
    --batchSize 1 \
    --gpu_ids 0 \
    --resize_or_crop resize_and_crop \
    --vgg 1 \
    --use_ragan \
    --skip 0 \
    --identity 5 > logs/train.log 2>&1 &