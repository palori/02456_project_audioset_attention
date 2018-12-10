#!/bin/bash

# SETUP FOLDERS TO RUN THE CODE #
echo "\n\n*********\n* Setup *\n*********\n"
cd ..

# 1. download 'packed_features'
#https://drive.google.com/file/d/0B49XSFgf-0yVQk01eG92RHg4WTA/view?fbclid=IwAR2jhbhIbKbjA-cVdk_lMeEHTt__yqjo8su3yjjFXlPfu5RisZ96L51ld8E
#echo "\n    1. Download dataset with features"
FEATURES="pf.zip"
#wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=0B49XSFgf-0yVQk01eG92RHg4WTA' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=0B49XSFgf-0yVQk01eG92RHg4WTA" -O $FEATURES && rm -rf /tmp/cookies.txt 

# extraxt and delete zip
#unzip $FEATURES
#rm $FEATURES

# 2. clone AudioSet repository from https://github.com/qiuqiangkong/audioset_classification
echo "\n    2. Clone AudioSet repository"
git clone https://github.com/qiuqiangkong/audioset_classification.git

# 3. clone DCASE repository
echo "\n    3. Clone DCASE repository"
git clone https://github.com/DCASE-REPO/dcase2018_baseline.git

#echo "\n    4. Download AudioSet csv's"
#mkdir audioset_csv
#cd audioset_csv
#wget storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv
#wget storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/balanced_train_segments.csv
#wget storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/eval_segments.csv
#cd ..

# 5. generate data files
#echo "\n    5. Generate dataset files"
#cd 02456_project_audioset_attention
#echo `pwd`
#sh `pwd`/data_generator.sh # call another shell script, not ready yet

# @@@@ Possibility to have our 'main2.py' and 'core2.py' in our repo and 
# copy/link it to 'audioset_classification'.
# then have a 'runme.sh' in our repo that will run the 'main2.py' from
# 'audioset_classification'


#### remove all
#echo "\nRemove all..."
#echo `pwd`
#sh `pwd`/remove_all.sh # call another shell script


echo "\n\n***************\n* Setup done! *\n***************\n"
