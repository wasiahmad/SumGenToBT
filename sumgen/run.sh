#!/usr/bin/env bash

export PYTHONIOENCODING=utf-8;
CURRENT_DIR=`pwd`
HOME_DIR=`realpath ..`;

GPU=${1:-0}
MODEL_SIZE=${2:-base};
export CUDA_VISIBLE_DEVICES=$GPU

languages=(java python);
lang_dict=$CURRENT_DIR/lang_dict.txt;
USER_DIR="$HOME_DIR/source";
PATH_2_DATA=${HOME_DIR}/data/sumgen/processed;
PRETRAIN=${HOME_DIR}/plbart/plbart_${MODEL_SIZE}.pt;
SAVE_DIR=$CURRENT_DIR/${MODEL_SIZE}_finetuned;
mkdir -p $SAVE_DIR

# effective bsz = 1024
BATCH_SIZE=8;
UPDATE_FREQ=16;
MAX_UPDATE=20000;
WARMUP=200;


function train() {

fairseq-train "$PATH_2_DATA"/binary \
    --fp16 \
    --user-dir $USER_DIR \
    --restore-file $PRETRAIN \
    --reset-dataloader \
    --reset-optimizer \
    --reset-meters \
    --reset-lr-scheduler \
    --task translation_multi_simple_epoch_extended \
    --lang-dict "$lang_dict" \
    --lang-pairs "$lang_pairs" \
    --batch-size $BATCH_SIZE \
    --update-freq $UPDATE_FREQ \
    --truncate-source \
    --arch mbart_${MODEL_SIZE} \
    --sampling-method "temperature" \
    --sampling-temperature 1.5 \
    --encoder-langtok "src" \
    --decoder-langtok \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --optimizer adam \
    --adam-eps 1e-06 \
    --adam-betas '(0.9, 0.98)' \
    --lr-scheduler inverse_sqrt \
    --lr 5e-05 \
    --warmup-updates $WARMUP \
    --max-update $MAX_UPDATE \
    --dropout 0.1 \
    --attention-dropout 0.1 \
    --weight-decay 0.1 \
    --eval-bleu \
    --eval-bleu-detok space \
    --eval-tokenized-bleu \
    --eval-bleu-remove-bpe sentencepiece \
    --eval-bleu-args '{"beam": 1}' \
    --best-checkpoint-metric bleu \
    --maximize-best-checkpoint-metric \
    --skip-invalid-size-inputs-valid-test \
    --save-interval-updates 1000 \
    --keep-interval-updates 1 \
    --save-interval 10000000 \
    --no-epoch-checkpoints \
    --patience 3 \
    --seed 1234 \
    --num-workers 4 \
    --log-format json \
    --log-interval 10 \
    --save-dir $SAVE_DIR \
    --valid-subset valid \
    2>&1 | tee $SAVE_DIR/training.log;

}


function evaluate_generation(){

SOURCE_LANG="en_XX";
TARGET_LANG=$1;
MODEL_PATH=${SAVE_DIR}/checkpoint_best.pt;
FILE_PREF=${SAVE_DIR}/${SOURCE_LANG}_${TARGET_LANG}_output;
RESULT_FILE=${SAVE_DIR}/${SOURCE_LANG}_${TARGET_LANG}_result.txt;
GOUND_TRUTH_PATH=${PATH_2_DATA}/test.${TARGET_LANG}-en_XX.${TARGET_LANG};

echo "==========================================================================" | tee ${RESULT_FILE};
echo "Source: ${SOURCE_LANG}                              Target: ${TARGET_LANG}" | tee -a ${RESULT_FILE};
echo "--------------------------------------------------------------------------" | tee -a ${RESULT_FILE};

fairseq-generate $PATH_2_DATA/binary \
    --path $MODEL_PATH \
    --user-dir $USER_DIR \
    --task translation_multi_simple_epoch_extended \
    --gen-subset test \
    --source-lang $SOURCE_LANG \
    --target-lang $TARGET_LANG \
    --remove-bpe 'sentencepiece'\
    --batch-size 16 \
    --encoder-langtok "src" \
    --decoder-langtok \
    --lang-dict $lang_dict \
    --lang-pairs $lang_pairs \
    --beam 5 > ${FILE_PREF};

cat $FILE_PREF | grep -P "^H" |sort -V |cut -f 3- | cut -d' ' -f 2- > $FILE_PREF.hyp;
if [[ "$(wc -l < ${FILE_PREF}.hyp)" -eq "$(wc -l < $GOUND_TRUTH_PATH)" ]]; then
    export PYTHONPATH=$HOME_DIR
    python ${HOME_DIR}/evaluation/pl_eval.py \
        --references ${GOUND_TRUTH_PATH} \
        --predictions ${FILE_PREF}.hyp \
        --detokenize \
        --lang $TARGET_LANG 2>&1 | tee ${RESULT_FILE};
else
    echo 'Warning: Number of predictions do not match the number of ground truth!' | tee -a ${RESULT_FILE};
fi

}


function evaluate_summarization(){

SOURCE_LANG="$1";
TARGET_LANG="en_XX";
MODEL_PATH=${SAVE_DIR}/checkpoint_best.pt;
FILE_PREF=${SAVE_DIR}/${SOURCE_LANG}_${TARGET_LANG}_output;
RESULT_FILE=${SAVE_DIR}/${SOURCE_LANG}_${TARGET_LANG}_result.txt;
GOUND_TRUTH_PATH=${PATH_2_DATA}/test.${SOURCE_LANG}-en_XX.en_XX;

echo "==========================================================================" | tee ${RESULT_FILE};
echo "Source: ${SOURCE_LANG}                              Target: ${TARGET_LANG}" | tee -a ${RESULT_FILE};
echo "--------------------------------------------------------------------------" | tee -a ${RESULT_FILE};

fairseq-generate ${PATH_2_DATA}/binary \
    --path $MODEL_PATH \
    --user-dir $USER_DIR \
    --task translation_multi_simple_epoch_extended \
    --gen-subset test \
    --source-lang $SOURCE_LANG \
    --target-lang $TARGET_LANG \
    --remove-bpe 'sentencepiece'\
    --batch-size 16 \
    --encoder-langtok "src" \
    --decoder-langtok \
    --lang-dict $lang_dict \
    --lang-pairs $lang_pairs \
    --beam 10 > ${FILE_PREF};

cat $FILE_PREF | grep -P "^H" |sort -V |cut -f 3- | cut -d' ' -f 2- > $FILE_PREF.hyp;
if [[ "$(wc -l < ${FILE_PREF}.hyp)" -eq "$(wc -l < $GOUND_TRUTH_PATH)" ]]; then
    python ${HOME_DIR}/evaluation/nl_eval.py \
        --references ${GOUND_TRUTH_PATH} \
        --predictions ${FILE_PREF}.hyp 2>&1 | tee -a ${RESULT_FILE};
else
    echo 'Warning: Number of predictions do not match the number of ground truth!' | tee -a ${RESULT_FILE};
fi

}

lang_pairs="";
for lang in ${languages[*]}; do
    lang_pairs=$lang_pairs"en_XX-$lang,$lang-en_XX,";
done
lang_pairs=${lang_pairs::-1}

train
for lang in ${languages[*]}; do
    evaluate_generation $lang;
    evaluate_summarization $lang;
done
