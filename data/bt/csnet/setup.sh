#!/usr/bin/env bash

CURRENT_DIR=`pwd`
HOME_DIR=`realpath ../../..`;

OUT_DIR=$CURRENT_DIR

function download () {

URL_PREFIX=https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2
for lang in java python; do
    if [[ ! -d $OUT_DIR/$lang ]]; then
        wget $URL_PREFIX/${lang}.zip -O ${lang}.zip
        unzip ${lang}.zip -d $OUT_DIR
        rm ${lang}.zip
        rm $OUT_DIR/${lang}_licenses.pkl
        mv $OUT_DIR/${lang}_dedupe_definitions_v2.pkl $OUT_DIR/$lang
        mv $OUT_DIR/$lang/final/*/*/* $OUT_DIR/$lang
        rm -rf $OUT_DIR/$lang/final
        cd $OUT_DIR/$lang
        for file in ./*.jsonl.gz; do
            gzip -d $file
        done
        cd $CURRENT_DIR
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


download
prepare
