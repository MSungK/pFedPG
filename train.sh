#! /usr/bin/zsh

output_dir="t"

# rm -r $output_dir
# mkdir $output_dir

for seed in "42" 
do
    echo "Current Seed: $seed"
    python pFedPG.py \
        --device 3 \
        --lr 1e-3 \
        --weight_decay 1e-4 \
        --server_epoch 5 \
        --config-file custom_configs/domainnet10.yaml \
        MODEL.TYPE "vit" \
        MODEL.PROMPT.NUM_TOKENS "10" \
        MODEL.PROMPT.DEEP "False" \
        MODEL.PROMPT.DROPOUT "0.1" \
        DATA.FEATURE "sup_vitb16_imagenet21k" \
        DATA.BATCH_SIZE "64" \
        DATA.NUM_WORKERS "8" \
        SOLVER.TOTAL_EPOCH "1" \
        SOLVER.BASE_LR "0.25" \
        SOLVER.WEIGHT_DECAY "0.001" \
        SEED $seed \
        MODEL.MODEL_ROOT "weights/" \
        OUTPUT_DIR $output_dir 
done