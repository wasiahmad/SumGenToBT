#!/usr/bin/env bash

CURRENT_DIR=`pwd`
HOME_DIR=`realpath ../..`;
export PYTHONIOENCODING=utf-8;

SPM_DIR=${HOME_DIR}/sentencepiece
DICT_FILE=${SPM_DIR}/dict.txt
SPM_VOCAB=${SPM_DIR}/sentencepiece.bpe.vocab
SPM_ENC_SCRIPT=${SPM_DIR}/encode.py

OUT_DIR=$CURRENT_DIR/eval_data;
SHARD_DIR=$CURRENT_DIR/eval_shard;
mkdir -p $OUT_DIR
mkdir -p $SHARD_DIR
cp $DICT_FILE $SHARD_DIR


function download_g4g_data () {

FILE=transcoder_test_set.zip
if [[ ! -f "$FILE" ]]; then
    wget https://dl.fbaipublicfiles.com/transcoder/test_set/$FILE
fi
unzip transcoder_test_set.zip
for split in valid test; do
    for langs in 'java-python' 'cpp-python' 'cpp-java'; do
        l1=$(echo "$langs" | cut -d- -f1)
        l2=$(echo "$langs" | cut -d- -f2)
        FILE=transcoder_${split}.${l1}.tok
        OUTFILE=${OUT_DIR}/$split.${l1}-${l2}.id
        cat test_dataset/$FILE | cut -d '|' -f 1 | awk '{$1=$1};1' > $OUTFILE
        OUTFILE=${OUT_DIR}/$split.${l1}-${l2}.$l1
        cat test_dataset/$FILE | cut -d '|' -f 2 | awk '{$1=$1};1' > $OUTFILE
        FILE=transcoder_${split}.${l2}.tok
        OUTFILE=${OUT_DIR}/$split.${l1}-${l2}.$l2
        cat test_dataset/$FILE | cut -d '|' -f 2 | awk '{$1=$1};1' > $OUTFILE
    done
done
rm -rf test_dataset

}


function spm_pl () {

for SPLIT in valid test; do
    for langs in 'java-python' 'cpp-python' 'cpp-java'; do
        l1=$(echo "$langs" | cut -d- -f1)
        l2=$(echo "$langs" | cut -d- -f2)
        python $SPM_ENC_SCRIPT \
            --model-file $SPM_DIR/sentencepiece.bpe.model \
            --inputs ${OUT_DIR}/$SPLIT.${l1}-${l2}.$l1 $SPLIT.${l1}-${l2}.$l2 \
            --outputs ${OUT_DIR}/$SPLIT.${l1}-${l2}.spm.$l1 $SPLIT.${l1}-${l2}.spm.$l2 \
            --max_len 510 \
            --workers 60;
    done
done

}


function binarize_pl () {

for langs in 'java-python' 'cpp-python' 'cpp-java'; do
    l1=$(echo "$langs" | cut -d- -f1)
    l2=$(echo "$langs" | cut -d- -f2)
    fairseq-preprocess \
        --source-lang $l1 \
        --target-lang $l2 \
        --validpref ${OUT_DIR}/valid.${l1}-${l2}.spm \
        --testpref ${OUT_DIR}/test.${l1}-${l2}.spm \
        --destdir $SHARD_DIR \
        --srcdict $DICT_FILE \
        --tgtdict $DICT_FILE \
        --workers 60;
done

}


download_g4g_data
spm_pl
binarize_pl
