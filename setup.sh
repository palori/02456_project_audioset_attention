#!/bin/bash

# SETUP FOLDERS TO RUN THE CODE #
echo "\n\n*********\n* Setup *\n*********\n"
cd test_data
# 1. download 'packed_features'
#https://drive.google.com/file/d/0B49XSFgf-0yVQk01eG92RHg4WTA/view?fbclid=IwAR2jhbhIbKbjA-cVdk_lMeEHTt__yqjo8su3yjjFXlPfu5RisZ96L51ld8E
# Permission denied!
#wget --no-check-certificate "https://drive.google.com/uc?export=download&id=0B49XSFgf-0yVQk01eG92RHg4WTA" -O "pf"
echo "\n    1. Download dataset with features"
FEATURES="pf.zip"
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=0B49XSFgf-0yVQk01eG92RHg4WTA' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=0B49XSFgf-0yVQk01eG92RHg4WTA" -O $FEATURES && rm -rf /tmp/cookies.txt 
#(template to use) wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=FILEID' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=FILEID" -O FILENAME && rm -rf /tmp/cookies.txt 

# extraxt and delete zip
unzip $FEATURES
rm $FEATURES

# 2. clone AudioSet repository from https://github.com/qiuqiangkong/audioset_classification
echo "\n    2. Clone AudioSet repository"
git clone https://github.com/qiuqiangkong/audioset_classification.git

# 3. clone DCASE repository
echo "\n    3. Clone DCASE repository"
git clone https://github.com/DCASE-REPO/dcase2018_baseline.git

# 4. generate data files
echo "\n    4. Generate dataset files"
cd 02456_project_audioset_attention
echo `pwd`
sh `pwd`/data_generator.sh # call another shell script



#### remove all
echo "\nRemove all..."
echo `pwd`
#sh `pwd`/remove_all.sh # call another shell script


echo "\n\n***************\n* Setup done! *\n***************\n"
