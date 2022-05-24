#!/usr/bin/env bash

export LC_ALL=C;
export PYTHONIOENCODING=utf-8;
CURRENT_DIR=`pwd`
CODE_DIR_HOME=`realpath ../..`;

SAVE_DIR=$1;
SOURCE=${2:-java};
TARGET=${3:-python};

export PYTHONPATH=$CODE_DIR_HOME
echo "Source: $SOURCE Target: $TARGET"

DATA_HOME_DIR=${CODE_DIR_HOME}/data;
evaluator_script="${CODE_DIR_HOME}/evaluation";
RESULT_FILE=${SAVE_DIR}/result.txt;
GOLD_PATH=${DATA_HOME_DIR}/bt/eval_data;

GOLD_SRC_FILE=${GOLD_PATH}/test.${SOURCE}-${TARGET}.$SOURCE;
GOLD_TGT_FILE=${GOLD_PATH}/test.${SOURCE}-${TARGET}.$TARGET;
GOLD_ID_FILE=${GOLD_PATH}/test.${SOURCE}-${TARGET}.id;
if [[ ! -f $GOLD_SRC_FILE ]]; then
    GOLD_SRC_FILE=${GOLD_PATH}/test.${TARGET}-${SOURCE}.$SOURCE;
    GOLD_TGT_FILE=${GOLD_PATH}/test.${TARGET}-${SOURCE}.$TARGET;
    GOLD_ID_FILE=${GOLD_PATH}/test.${TARGET}-${SOURCE}.id;
fi

OUT_FOLDER=$SAVE_DIR/scripts;
mkdir -p $OUT_FOLDER;
FILE_PREF=${SAVE_DIR}/test;

python $evaluator_script/pl_eval.py \
    --references $GOLD_TGT_FILE \
    --predictions $FILE_PREF.output \
    --lang $TARGET \
    2>&1 | tee $RESULT_FILE;

python $evaluator_script/compute_ca.py \
    --src_path $GOLD_SRC_FILE \
    --ref_path $GOLD_TGT_FILE \
    --id_path $GOLD_ID_FILE \
    --hyp_paths $FILE_PREF.output \
    --split test \
    --outfolder $OUT_FOLDER \
    --source_lang $SOURCE \
    --target_lang $TARGET \
    --retry_mismatching_types True \
    2>&1 | tee -a $RESULT_FILE;

python $evaluator_script/classify_errors.py \
    --logfile ${SAVE_DIR}/hyp.${SOURCE}-${TARGET}.test_beam0.out.txt \
    --lang $TARGET \
    2>&1 | tee -a $RESULT_FILE;
