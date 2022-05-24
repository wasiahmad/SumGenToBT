#!/usr/bin/env bash

export PYTHONIOENCODING=utf-8;
CURRENT_DIR=`pwd`
CODE_DIR_HOME=`realpath ..`;

GPU_ID=$1
SPLIT=${2:-test}
MODEL_SIZE=${3:-base}

BEAMS=(1 10 5 10)
NBEST=(1 1 5 10)

function run () {

SRC=$1
TGT=$2

for ((i=0; i<${#BEAMS[*]}; ++i)); do
    beam=${BEAMS[$i]}
    top=${NBEST[$i]}
    SAVE_DIR=${CURRENT_DIR}/${SRC}2${TGT}_b${beam}_t${top};
    MODEL_DIR=${CODE_DIR_HOME}/plbart/bt_${MODEL_SIZE}_java_python;
    bash decode.sh $GPU_ID $SRC $TGT $beam $top $SPLIT $MODEL_DIR;
    bash evaluate.sh $SAVE_DIR $SRC $TGT $top $SPLIT;
done

}

run java python
run python java
