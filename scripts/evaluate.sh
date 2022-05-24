#!/usr/bin/env bash

export LC_ALL=C;
export PYTHONIOENCODING=utf-8;
CURRENT_DIR=`pwd`
CODE_DIR_HOME=`realpath ..`;

SAVE_DIR=$1;
SOURCE=${2:-java};
TARGET=${3:-python};
NBEST=${4:-1};
SPLIT=${5:-test};

export PYTHONPATH=$CODE_DIR_HOME
echo "Source: $SOURCE Target: $TARGET"

DATA_HOME_DIR=${CODE_DIR_HOME}/data;
evaluator_script="${CODE_DIR_HOME}/evaluation";
RESULT_FILE=${SAVE_DIR}/result_${SPLIT}_${NBEST}.txt;
GOLD_PATH=${DATA_HOME_DIR}/bt/eval_data;

GOLD_SRC_FILE=${GOLD_PATH}/${SPLIT}.${SOURCE}-${TARGET}.$SOURCE;
GOLD_TGT_FILE=${GOLD_PATH}/${SPLIT}.${SOURCE}-${TARGET}.$TARGET;
GOLD_ID_FILE=${GOLD_PATH}/${SPLIT}.${SOURCE}-${TARGET}.id;
if [[ ! -f $GOLD_SRC_FILE ]]; then
    GOLD_SRC_FILE=${GOLD_PATH}/${SPLIT}.${TARGET}-${SOURCE}.$SOURCE;
    GOLD_TGT_FILE=${GOLD_PATH}/${SPLIT}.${TARGET}-${SOURCE}.$TARGET;
    GOLD_ID_FILE=${GOLD_PATH}/${SPLIT}.${TARGET}-${SOURCE}.id;
fi

OUT_FOLDER=$SAVE_DIR/${SPLIT}_scripts;
mkdir -p $OUT_FOLDER;

FILE_PREF=${SAVE_DIR}/${SPLIT};
PRED_FILES=()
for ((i=0;i<$NBEST;i++)); do
    PRED_FILES+=($FILE_PREF.$i.output)
done

python $evaluator_script/pl_eval.py \
    --references $GOLD_TGT_FILE \
    --predictions "${PRED_FILES[@]}" \
    --lang $TARGET \
    2>&1 | tee $RESULT_FILE;

python $evaluator_script/compute_ca.py \
    --src_path $GOLD_SRC_FILE \
    --ref_path $GOLD_TGT_FILE \
    --id_path $GOLD_ID_FILE \
    --hyp_paths "${PRED_FILES[@]}" \
    --split $SPLIT \
    --outfolder $OUT_FOLDER \
    --source_lang $SOURCE \
    --target_lang $TARGET \
    --retry_mismatching_types True \
    2>&1 | tee -a $RESULT_FILE;

python $evaluator_script/classify_errors.py \
    --logfile ${SAVE_DIR}/hyp.${SOURCE}-${TARGET}.${SPLIT}_beam0.out.txt \
    --lang $TARGET \
    2>&1 | tee -a $RESULT_FILE;
