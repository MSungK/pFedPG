#! /usr/bin/zsh

rm -r results/*

for seed in "42" "44" "46" "48" "50"
do
    echo "Current Seed: $seed"
    python pFedPG.py \
        --device 1 \
        --lr 1e-3 \
        --weight_decay 1e-4 \
        --server_epoch 5 \
        --config-file custom_configs/office_caltech10.yaml \
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
        OUTPUT_DIR "results" 
done