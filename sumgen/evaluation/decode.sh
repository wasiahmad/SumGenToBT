#!/usr/bin/env bash

export PYTHONIOENCODING=utf-8;
CURRENT_DIR=`pwd`
CODE_DIR_HOME=`realpath ../..`;

GPU=${1:-0};
SOURCE=${2:-java};
TARGET=${3:-python};
MODEL_SIZE=${4:-base};
BEAM=${5:-10};

export PYTHONPATH=$CODE_DIR_HOME
export CUDA_VISIBLE_DEVICES=$GPU
echo "Source: $SOURCE Target: $TARGET"

DATA_HOME_DIR=${CODE_DIR_HOME}/data;
USER_DIR=${CODE_DIR_HOME}/source;
SAVE_DIR=${CURRENT_DIR}/${MODEL_SIZE}_${SOURCE}2${TARGET}_b${BEAM};
mkdir -p $SAVE_DIR

evaluator_script="${CODE_DIR_HOME}/evaluation";
restore_path=${CODE_DIR_HOME}/sumgen/${MODEL_SIZE}_finetuned/checkpoint_best.pt;
SPM_DIR=${CODE_DIR_HOME}/sentencepiece
SPM_ENC_SCRIPT=${SPM_DIR}/encode.py

languages=(java python)
lang_pairs="";
for lang in ${languages[*]}; do
    lang_pairs=$lang_pairs"en_XX-$lang,$lang-en_XX,";
done
lang_pairs=${lang_pairs::-1}
lang_dict=${CODE_DIR_HOME}/sumgen/lang_dict.txt;

BIN_DIR=$SAVE_DIR/binary;

if [[ ! -d $$BIN_DIR ]]; then
    mkdir -p $BIN_DIR
    SHARD_DIR=$DATA_HOME_DIR/bt_shards
    if [[ -f $SHARD_DIR/test.$SOURCE-$TARGET.$SOURCE.bin ]]; then
        for ext in bin idx; do
            cp $SHARD_DIR/test.$SOURCE-$TARGET.$SOURCE.$ext $BIN_DIR/test.$SOURCE-en_XX.$SOURCE.$ext;
        done
    else
        for ext in bin idx; do
            cp $SHARD_DIR/test.$TARGET-$SOURCE.$SOURCE.$ext $BIN_DIR/test.$SOURCE-en_XX.$SOURCE.$ext;
        done
    fi
    cp $SHARD_DIR/dict.txt $BIN_DIR/dict.$SOURCE.txt;
    cp $SHARD_DIR/dict.txt $BIN_DIR/dict.en_XX.txt;
fi


########################## From PL to NL ##########################
FILE_PREF=${SAVE_DIR}/test_en_XX;

fairseq-generate $BIN_DIR \
    --user-dir $USER_DIR \
    --path $restore_path \
    --task translation_multi_simple_epoch_extended \
    --gen-subset test \
    --source-lang $SOURCE \
    --target-lang en_XX \
    --remove-bpe 'sentencepiece' \
    --batch-size 16 \
    --encoder-langtok "src" \
    --decoder-langtok \
    --lang-dict $lang_dict \
    --lang-pairs $lang_pairs \
    --max-len-b 500 \
    --beam $BEAM \
    --nbest 1 \
    --seed 1234 > $FILE_PREF.log;

cat $FILE_PREF.log | grep -P "^H" |sort -V |cut -f 3- > $FILE_PREF.output;


########################## From NL to PL ##########################
python $SPM_ENC_SCRIPT \
    --model-file $SPM_DIR/sentencepiece.bpe.model \
    --inputs $FILE_PREF.output \
    --outputs $FILE_PREF.output.spm.en_XX \
    --max_len 510 \
    --workers 60;

cp ${SPM_DIR}/dict.txt $SAVE_DIR
cp ${SPM_DIR}/dict.txt $SAVE_DIR/dict.$TARGET.txt

fairseq-preprocess \
    --source-lang en_XX \
    --target-lang $TARGET \
    --only-source \
    --testpref $FILE_PREF.output.spm \
    --destdir $SAVE_DIR \
    --srcdict $SAVE_DIR/dict.txt \
    --workers 60;

FILE_PREF=${SAVE_DIR}/test;

fairseq-generate $SAVE_DIR \
    --user-dir $USER_DIR \
    --path $restore_path \
    --task translation_multi_simple_epoch_extended \
    --gen-subset test \
    --source-lang en_XX \
    --target-lang $TARGET \
    --remove-bpe 'sentencepiece' \
    --encoder-langtok "src" \
    --decoder-langtok \
    --lang-dict $lang_dict \
    --lang-pairs $lang_pairs \
    --max-len-b 500 \
    --batch-size 8 \
    --beam $BEAM \
    --nbest 1 \
    --seed 1234 > $FILE_PREF.log;

cat $FILE_PREF.log | grep -P "^H" |sort -V |cut -f 3- > $FILE_PREF.output;
