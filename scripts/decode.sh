#!/usr/bin/env bash

export PYTHONIOENCODING=utf-8;
CURRENT_DIR=`pwd`
CODE_DIR_HOME=`realpath ..`;

GPU=${1:-0};
SOURCE=${2:-java};
TARGET=${3:-python};
BEAM=${4:-1};
NBEST=${5:-1};
SPLIT=${6:-test};
MODEL_DIR=${7:-"${CODE_DIR_HOME}/plbart/bt_base_java_python"};

export PYTHONPATH=$CODE_DIR_HOME
export CUDA_VISIBLE_DEVICES=$GPU
echo "Source: $SOURCE Target: $TARGET"

DATA_HOME_DIR=${CODE_DIR_HOME}/data;
USER_DIR=${CODE_DIR_HOME}/source;
SAVE_DIR=${CURRENT_DIR}/${SOURCE}2${TARGET}_b${BEAM}_t${NBEST};
mkdir -p $SAVE_DIR

evaluator_script="${CODE_DIR_HOME}/evaluation";
langs=java,python,en_XX
checkpoint_file=checkpoint_best.pt
restore_path=${MODEL_DIR}/$checkpoint_file;
FILE_PREF=${SAVE_DIR}/${SPLIT};

fairseq-generate $DATA_HOME_DIR/bt/eval_shard \
    --user-dir $USER_DIR \
    --langs $langs \
    --path $restore_path \
    --task backtranslation \
    --gen-subset $SPLIT \
    --source-lang $SOURCE \
    --target-lang $TARGET \
    --remove-bpe 'sentencepiece' \
    --max-len-b 500 \
    --batch-size 8 \
    --beam $BEAM \
    --nbest $NBEST \
    --seed 1234 > $FILE_PREF.log;

cat $FILE_PREF.log | grep -P "^H" |sort -V |cut -f 3- > $FILE_PREF.output;
for ((i=0;i<$NBEST;i++)); do
    awk "NR % $NBEST == $i" $FILE_PREF.output > $FILE_PREF.$(($NBEST-$i-1)).output
done
