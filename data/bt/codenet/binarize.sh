#!/usr/bin/env bash

export PYTHONIOENCODING=utf-8;
CURRENT_DIR=`pwd`
HOME_DIR=`realpath ../../..`;

SPM_DIR=${HOME_DIR}/sentencepiece
DICT_FILE=${SPM_DIR}/dict.txt
SPM_VOCAB=${SPM_DIR}/sentencepiece.bpe.vocab
SPM_ENC_SCRIPT=${SPM_DIR}/encode.py

DATA_DIR=$CURRENT_DIR
SHARD_DIR=${CURRENT_DIR}/shard
mkdir -p $SHARD_DIR
cp $DICT_FILE $SHARD_DIR


function spm_pl () {

for LANG in java python; do
    if [[ ! -f $DATA_DIR/$LANG/train.functions.spm ]]; then
        python $SPM_ENC_SCRIPT \
            --model-file $SPM_DIR/sentencepiece.bpe.model \
            --inputs $DATA_DIR/$LANG/train.functions.tok \
            --outputs $DATA_DIR/$LANG/train.functions.spm \
            --max_len 510 \
            --strict_max_len \
            --workers 60;
    fi
done

}


function binarize_pl () {

for LANG in java python; do
    if [[ ! -d $SHARD_DIR/$LANG ]]; then
        fairseq-preprocess \
            --only-source \
            --trainpref $DATA_DIR/$LANG/train.functions.spm \
            --destdir $SHARD_DIR/$LANG \
            --srcdict $DICT_FILE \
            --workers 60;
    fi
done

}

spm_pl
binarize_pl
# copy all shard files for the eval data
cp -r ${HOME_DIR}/data/bt/eval_shard/* $SHARD_DIR
