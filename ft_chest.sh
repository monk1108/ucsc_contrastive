torchrun --nproc_per_node=2 --master_port 14285 chest_class_finetuning.py \
--batch_size 24 --input_size 256 --eval \
    --epochs 100 --warmup_epochs 20 \
    --model beit_large_patch16_224 --nb_classes 14 \
    --imagenet_default_mean_and_std \
    --model_key state_dict --model_prefix module.visual. \
    --disable_rel_pos_bias --abs_pos_emb --use_cls \
    --layer_scale_init_value 0 \
    --mixup 0 --cutmix 0 \
    --lr 0.0005 --drop_path 0.1 --layer_decay 0.95 \
    --log_dir ./myoutput/811/256bs48log/ --smoothing 0 \
    --output_dir ./myoutput/811/256bs48output/ \
    --finetune ./SLIP/checkpoint/clip_large_25ep.pt

