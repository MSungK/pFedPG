#! /usr/bin/zsh

rm -r results/*

python sample.py \
    --config-file configs/prompt/dogs.yaml \
    MODEL.TYPE "vit" \
    DATA.BATCH_SIZE "64" \
    MODEL.PROMPT.NUM_TOKENS "10" \
    MODEL.PROMPT.DEEP "False" \
    MODEL.PROMPT.DROPOUT "0.1" \
    DATA.FEATURE "sup_vitb16_imagenet21k" \
    DATA.NAME "StanfordDogs" \
    DATA.NUMBER_CLASSES "120" \
    SOLVER.BASE_LR "0.001" \
    SOLVER.WEIGHT_DECAY "0.0001" \
    SEED "42" \
    MODEL.MODEL_ROOT "weights/" \
    DATA.DATAPATH "dataset/standford_dogs" \
    OUTPUT_DIR "results"