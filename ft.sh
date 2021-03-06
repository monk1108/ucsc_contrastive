torchrun --nproc_per_node=4 run_class_finetuning.py \
--batch_size 128  --save_ckpt --save_ckpt_freq 10 \
    --epochs 100 --warmup_epochs 20 \
    --model beit_small_patch16_224 --nb_classes 1000 \
    --imagenet_default_mean_and_std \
    --model_key state_dict --model_prefix module.visual. \
    --disable_rel_pos_bias --abs_pos_emb --use_cls \
    --mixup 0.8 --cutmix 1 \
    --layer_scale_init_value 0 \
    --lr 4e-3 --drop_path 0 --layer_decay 0.65 \
    --output_dir ./myoutput/small --log_dir ./mylog/small \
    --finetune ./SLIP/checkpoint/clip_small_25ep.pt
