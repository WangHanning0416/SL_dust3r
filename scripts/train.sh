TRAIN_DATASET="4000 @ ScanNetpp(split='train', ROOT='/data/yuzheng/data/scannetpp_v2/scannetpp_processed', aug_crop=16, resolution=224, transform=ColorJitter) + \
4000 @ ARKitScenes(split='train',ROOT='/nvme/data/jiaheng/dust3r/arkitscenes_preprocessed', aug_crop=16, resolution=224, transform=ColorJitter)"

TEST_DATASET="400 @ ScanNetpp(split='test', ROOT='/data/yuzheng/data/scannetpp_v2/scannetpp_processed', resolution=224, seed=777)"

torchrun --nproc_per_node=4 train.py \
    --train_dataset "${TRAIN_DATASET}" \
    --test_dataset "${TEST_DATASET}" \
    --model "AsymmetricCroCo3DStereo(pos_embed='RoPE100', img_size=(224, 224), head_type='linear', output_mode='pts3d', depth_mode=('exp', -inf, inf), conf_mode=('exp', 1, inf), enc_embed_dim=1024, enc_depth=24, enc_num_heads=16, dec_embed_dim=768, dec_depth=12, dec_num_heads=12)" \
    --train_criterion "ConfLoss(Regr3D(L21, norm_mode='avg_dis'), alpha=0.2)" \
    --test_criterion "Regr3D_ScaleShiftInv(L21, gt_scale=True)" \
    --pretrained "/data/hanning/SL_dust3r/checkpoints/DUSt3R_ViTLarge_BaseDecoder_224_linear.pth" \
    --lr 0.0001 --min_lr 1e-06 --warmup_epochs 4 --epochs 60 --batch_size 16 --accum_iter 1 \
    --save_freq 5 --keep_freq 10 --eval_freq 1 \
    --output_dir "/nvme/data/hanning/checkpoints/dust3r_kinectsp_224_inject_ray"	  