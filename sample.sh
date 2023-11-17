#! /usr/bin/zsh

rm -r results/*

python pFedPG.py \
    --device 1 \
    --config-file custom_configs/office_caltech10.yaml \
    MODEL.TYPE "vit" \
    DATA.BATCH_SIZE "8" \
    MODEL.PROMPT.NUM_TOKENS "10" \
    MODEL.PROMPT.DEEP "False" \
    MODEL.PROMPT.DROPOUT "0.1" \
    DATA.FEATURE "sup_vitb16_imagenet21k" \
    SOLVER.TOTAL_EPOCH "1" \
    SOLVER.BASE_LR "0.25" \
    SOLVER.WEIGHT_DECAY "0.001" \
    SEED "42" \
    MODEL.MODEL_ROOT "weights/" \
    OUTPUT_DIR "results" 