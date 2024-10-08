#! /usr/bin/zsh

output_dir="results/domain_pFedPG"
# ratio=0.05
# rm -r $output_dir
# mkdir $output_dir

for ratio in "0.02" "0.01" 
do
    echo "Current Ratio: $ratio"
    for seed in "44" 
    do
        echo "Current Seed: $seed"
        python pFedPG.py \
            --device 3 \
            --lr 1e-3 \
            --weight_decay 1e-6 \
            --server_epoch 20 \
            --config-file custom_configs/domainnet10.yaml \
            MODEL.TYPE "vit" \
            MODEL.PROMPT.NUM_TOKENS "10" \
            MODEL.PROMPT.DEEP "False" \
            MODEL.PROMPT.DROPOUT "0.1" \
            DATA.FEATURE "sup_vitb16_imagenet21k" \
            DATA.BATCH_SIZE "64" \
            DATA.NUM_WORKERS "4" \
            DATA.TRAIN_RATIO $ratio \
            SOLVER.TOTAL_EPOCH "5" \
            SOLVER.BASE_LR "0.25" \
            SOLVER.WEIGHT_DECAY "0.001" \
            SEED $seed \
            MODEL.MODEL_ROOT "weights/" \
            OUTPUT_DIR "$output_dir-$ratio"
    done
done