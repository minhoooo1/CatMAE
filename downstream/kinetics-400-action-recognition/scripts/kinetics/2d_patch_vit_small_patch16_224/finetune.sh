# Set the path to save checkpoints
OUTPUT_DIR='output/catmae_vit_small_patch16_224/eval_lr_1.2e-3_ep_150'
# path to Kinetics set (train.csv/val.csv/test.csv)
DATA_PATH='/path/to/data/Kinetics-400'
# path to pretrain model
MODEL_PATH='checkpoint/catmae-vit-s-16-300ep.pth'

# We add repeated_aug (--num_sample = 2) on Kinetics-400 here, 
# which could better performance while need more time for fine-tuning

# batch_size can be adjusted according to number of GPUs
# this script is for 32 GPUs (4 nodes x 8 GPUs)
python run_class_finetuning.py \
    --model vit_small_patch16_224 \
    --patchify_type 2d \
    --data_set Kinetics-400 \
    --nb_classes 400 \
    --data_path ${DATA_PATH} \
    --finetune ${MODEL_PATH} \
    --log_dir ${OUTPUT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size 7 \
    --num_sample 2 \
    --input_size 224 \
    --short_side_size 224 \
    --save_ckpt_freq 10 \
    --num_frames 16 \
    --sampling_rate 4 \
    --opt adamw \
    --lr 1.2e-3 \
    --opt_betas 0.9 0.999 \
    --weight_decay 0.05 \
    --epochs 150 \
    --test_num_segment 5 \
    --test_num_crop 3 \
    --dist_eval \
    --gpus "0,1,2,3"
