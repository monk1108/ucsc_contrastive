torchrun --nproc_per_node=8 --master_port 12123 chest_class_finetuning.py \
--batch_size 128  \
    --epochs 100 --warmup_epochs 20 \
    --model beit_base_patch16_224 --nb_classes 14 \
    --imagenet_default_mean_and_std \
    --model_key state_dict --model_prefix module.visual. \
    --disable_rel_pos_bias --abs_pos_emb --use_cls \
    --layer_scale_init_value 0 \
    --mixup 0.95 --cutmix 1 \
    --lr 1e-4 --drop_path 0.1 --layer_decay 0.95 \
    --log_dir ./myoutput/mmlr4Mixup95log/ --smoothing 0 \
    --output_dir ./myoutput/mmlr4Mixup95output/ \
    --finetune ./SLIP/checkpoint/clip_base_25ep.pt

