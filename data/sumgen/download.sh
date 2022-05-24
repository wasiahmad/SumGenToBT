#!/usr/bin/env bash

#############################################
#          Download CSNET dataset           #
#############################################

wget https://raw.githubusercontent.com/microsoft/CodeXGLUE/main/Code-Text/code-to-text/dataset.zip
unzip dataset.zip
mv dataset/* .

wget https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/python.zip
wget https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/java.zip

unzip python.zip
unzip java.zip
rm *.zip
rm *.pkl

python preprocess.py
rm -r */final
rm -r dataset
rm preprocess.py
rm -rf go php javascript ruby

#############################################
#          Download CCSD dataset            #
#############################################

mkdir -p cpp;
cd cpp;

FILE=c_functions_all_data.jsonl.gz
fileid="1cGb5GSBxOd8u4Zxn63K4DPaRARu1heJz"
curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o ${FILE}
gzip -d ${FILE}
head -n 90000 c_functions_all_data.jsonl > train.jsonl
head -n 2500 c_functions_all_data.jsonl > valid.jsonl
head -n 2781 c_functions_all_data.jsonl > test.jsonl
rm c_functions_all_data.jsonl

FILE=all_functions.csv
fileid="1pbjM5uGZFAYQLRRF3KVk1EE32x6uWXib"
curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o ${FILE}

rm ./cookie
cd ..;
