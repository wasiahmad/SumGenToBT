#!/usr/bin/env bash

CURRENT_DIR=`pwd`
HOME_DIR=`realpath ../../..`;
OUT_DIR=$CURRENT_DIR


function extract () {

for lang in java python; do
    mkdir -p $OUT_DIR/$lang
    if [[ ! -f $OUT_DIR/$lang/${lang}_train_0.jsonl ]]; then
        python extract.py \
            --lang $lang \
            --source_dir $CURRENT_DIR/Project_CodeNet \
            --target_dir $OUT_DIR/$lang \
            --category all;
    fi
done

}


function prepare () {

for lang in java python; do
    for split in train; do
        if [[ ! -f $OUT_DIR/$lang/${split}.functions.tok ]]; then
            export PYTHONPATH=$HOME_DIR;
            python preprocess.py \
                --lang $lang \
                --split $split \
                --source_dir $OUT_DIR \
                --target_pl_dir $OUT_DIR/$lang \
                --root_folder ${HOME_DIR}/third_party \
                --keep_standalone_only \
                --workers 60;
        fi
    done
done

}


extract
prepare
