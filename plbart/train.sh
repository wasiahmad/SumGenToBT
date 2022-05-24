#!/usr/bin/env bash

CURRENT_DIR=`pwd`
HOME_DIR=`realpath ..`;

GPU=${1:-0};
MODEL_SIZE=${2:-base};

DATA_HOME=${HOME_DIR}/data
CODENET_DATA_HOME=${DATA_HOME}/bt/codenet
CSNET_DATA_HOME=${DATA_HOME}/bt/csnet
SPM_MODEL=${HOME_DIR}/sentencepiece/sentencepiece.bpe.model
langs=java,python,en_XX
mono_langs=java,python
valid_lang_pairs='java-python'

# use the following if we use BigQuery dataset
# LARGE_DATA_HOME=?
# DATA_DIR=""
# for (( idx=0; idx<=7; idx++ )); do
#     DATA_DIR+="${LARGE_DATA_HOME}/bt_shards/shard${idx}"
#     if [[ $idx < 7 ]]; then
#         DATA_DIR+=":"
#     fi
# done

DATA_DIR=${CSNET_DATA_HOME}/shard
USER_DIR=${HOME_DIR}/source
SAVE_DIR=${CURRENT_DIR}/bt_${MODEL_SIZE}_${mono_langs//,/_}
mkdir -p $SAVE_DIR
TENSORBOARD_LOGDIR=${SAVE_DIR}/tensorboard_logs

ADDITIONAL_PARAMS=""
if [[ ! -f $SAVE_DIR/checkpoint_last.pt ]]; then
    PRETRAIN=${HOME_DIR}/sumgen/${MODEL_SIZE}_finetuned/checkpoint_best.pt
    ADDITIONAL_PARAMS="--restore-file $PRETRAIN "
    ADDITIONAL_PARAMS+="--reset-optimizer "
    ADDITIONAL_PARAMS+="--reset-meters "
    ADDITIONAL_PARAMS+="--reset-dataloader "
    ADDITIONAL_PARAMS+="--reset-lr-scheduler "
fi

export LC_ALL=C
export CUDA_VISIBLE_DEVICES=$GPU


function pretrain () {

# effective batch size = 1024
MAX_UPDATE=10000
WARMUP_UPDATES=100
INTER_TRANS_STEPS=200
BATCH_SIZE=8
UPDATE_FREQ=8

fairseq-train $DATA_DIR $ADDITIONAL_PARAMS \
    --user-dir $USER_DIR \
    --langs $langs \
    --mono-langs $mono_langs \
    --inter-lang en_XX \
    --inter-trans-steps $INTER_TRANS_STEPS \
    --valid-lang-pairs $valid_lang_pairs \
    --dataset-impl 'mmap' \
    --bpe 'sentencepiece' \
    --sentencepiece-model $SPM_MODEL \
    --arch mbart_${MODEL_SIZE} \
    --batch-size $BATCH_SIZE \
    --update-freq $UPDATE_FREQ \
    --layernorm-embedding \
    --train-subset train \
    --valid-subset valid \
    --required-batch-size-multiple 8 \
    --task backtranslation \
    --criterion cross_entropy \
    --dropout 0.1 \
    --attention-dropout 0.1 \
    --relu-dropout 0.0 \
    --weight-decay 0.01 \
    --optimizer adam \
    --adam-eps 1e-06 \
    --clip-norm 0.1 \
    --lr 3e-4 \
    --lr-scheduler polynomial_decay \
    --warmup-updates $WARMUP_UPDATES \
    --total-num-update $MAX_UPDATE \
    --max-update $MAX_UPDATE \
    --show-samples-interval 1000 \
    --fp16 \
    --ddp-backend=no_c10d \
    --save-interval-updates 100 \
    --save-dir $SAVE_DIR \
    --validate-interval 10000000 \
    --save-interval 10000000 \
    --no-epoch-checkpoints \
    --skip-invalid-size-inputs-valid-test \
    --log-format json \
    --log-interval 10 \
    --num-workers 4 \
    --seed 1234 \
    --eval-bleu \
    --eval-bleu-detok space \
    --eval-tokenized-bleu \
    --eval-bleu-remove-bpe sentencepiece \
    --eval-bleu-args '{"beam": 5}' \
    --eval-bleu-print-samples \
    --show-eval-samples-interval 10 \
    --best-checkpoint-metric bleu \
    --maximize-best-checkpoint-metric \
    --patience 10 \
    --tensorboard-logdir $TENSORBOARD_LOGDIR \
    2>&1 | tee $SAVE_DIR/output.log;

}


function select_best_checkpoint () {

SOURCE=$1;
TARGET=$2;

GOLD_PATH=${DATA_HOME}/bt/eval_data;
GOLD_ID_FILE=${GOLD_PATH}/valid.${SOURCE}-${TARGET}.id;
GOLD_TGT_FILE=${GOLD_PATH}/valid.${SOURCE}-${TARGET}.$TARGET;
if [[ ! -f $GOLD_ID_FILE ]]; then
    GOLD_ID_FILE=${GOLD_PATH}/valid.${TARGET}-${SOURCE}.id;
    GOLD_TGT_FILE=${GOLD_PATH}/valid.${TARGET}-${SOURCE}.$TARGET;
fi

IGNORE_CKPTS=(checkpoint_best checkpoint_last)
OUT_FOLDER=$SAVE_DIR/scripts;
RESULT_FILE=${SAVE_DIR}/validation_results_${SOURCE}2${TARGET}.txt;

for ckpt_file in ${SAVE_DIR}/*.pt; do
    filename="$(basename -s .pt ${ckpt_file})"
    if [[ ! " ${IGNORE_CKPTS[*]} " =~ " ${filename} " ]]; then
        echo "validating ${filename}";
        FILE_PREF=${SAVE_DIR}/${SOURCE}_${TARGET}_${filename};
        fairseq-generate $DATA_HOME/bt_shards \
            --user-dir $USER_DIR \
            --langs $langs \
            --path $ckpt_file \
            --task backtranslation \
            --gen-subset valid \
            --source-lang $SOURCE \
            --target-lang $TARGET \
            --remove-bpe 'sentencepiece' \
            --max-len-b 256 \
            --batch-size 32 \
            --beam 1 \
            --seed 1234 > $FILE_PREF.log;
        cat $FILE_PREF.log | grep -P "^H" |sort -V |cut -f 3- > $FILE_PREF.output;
        rm $FILE_PREF.log;
    fi
done

export PYTHONPATH=$HOME_DIR;
python selector.py \
    --ref_path $GOLD_TGT_FILE \
    --id_path $GOLD_ID_FILE \
    --hyp_dir $SAVE_DIR \
    --split valid \
    --outfolder $OUT_FOLDER \
    --source_lang $SOURCE \
    --target_lang $TARGET \
    --retry_mismatching_types True \
    2>&1 | tee $RESULT_FILE;

}

pretrain
# select_best_checkpoint java python
# select_best_checkpoint python java
