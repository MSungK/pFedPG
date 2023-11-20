#! /usr/bin/zsh

for seed in "42" 
do
    echo "Current Seed: $seed"
    python single_vpt.py \
        --device 3 \
        --lr 1e-3 \
        --weight_decay 1e-4 \
        --server_epoch 100 \
        --config-file custom_configs/domainnet10.yaml \
        MODEL.TYPE "vit" \
        DATA.BATCH_SIZE "8" \
        MODEL.PROMPT.NUM_TOKENS "10" \
        MODEL.PROMPT.DEEP "False" \
        MODEL.PROMPT.DROPOUT "0.1" \
        DATA.FEATURE "sup_vitb16_imagenet21k" \
        DATA.BATCH_SIZE "64" \
        SOLVER.TOTAL_EPOCH "5" \
        SOLVER.BASE_LR "0.25" \
        SOLVER.WEIGHT_DECAY "0.001" \
        SEED $seed \
        MODEL.MODEL_ROOT "weights/" \
        OUTPUT_DIR "single_vpt" 
done