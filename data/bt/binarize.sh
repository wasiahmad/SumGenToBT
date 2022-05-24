#!/usr/bin/env bash

export PYTHONIOENCODING=utf-8;
CURRENT_DIR=`pwd`
HOME_DIR=`realpath ../..`;

SPM_DIR=${HOME_DIR}/sentencepiece
DICT_FILE=${SPM_DIR}/dict.txt
SPM_VOCAB=${SPM_DIR}/sentencepiece.bpe.vocab
SPM_ENC_SCRIPT=${SPM_DIR}/encode.py

DATA_DIR=? # fill-this-variable
SHARD_DIR=${CURRENT_DIR}/bt_shards
mkdir -p $SHARD_DIR

for (( idx=0; idx<=7; idx++ )); do
    mkdir -p ${SHARD_DIR}/shard${idx}
    cp $DICT_FILE ${SHARD_DIR}/shard${idx}
done


function spm_pl () {

for LANG in java python cpp; do
    for i in $(seq 0 7); do
        if [[ ! -f $DATA_DIR/$LANG/train.$i.functions_standalone.spm ]]; then
            python $SPM_ENC_SCRIPT \
                --model-file $SPM_DIR/sentencepiece.bpe.model \
                --inputs $DATA_DIR/$LANG/train.$i.functions_standalone.tok \
                --outputs $DATA_DIR/$LANG/train.$i.functions_standalone.spm \
                --max_len 256 \
                --strict_max_len \
                --workers 60;
        fi
    done
done

}


function binarize_pl () {

for LANG in java python cpp; do
    for i in $(seq 0 7); do
        LANG_SHARD=$SHARD_DIR/shard${i}/$LANG
        if [[ ! -d $LANG_SHARD ]]; then
            fairseq-preprocess \
                --only-source \
                --trainpref $DATA_DIR/$LANG/train.$i.functions_standalone.spm \
                --destdir $LANG_SHARD \
                --srcdict $DICT_FILE \
                --workers 60;
        fi
    done
done

}

spm_pl
binarize_pl
