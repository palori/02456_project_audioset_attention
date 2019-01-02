#!/bin/bash

# SETUP FOLDERS TO RUN THE CODE #
echo
echo
echo "*********"
echo "* Setup *"
echo "*********"
echo
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
echo
echo "    2. Clone AudioSet repository"
git clone https://github.com/qiuqiangkong/audioset_classification.git

# 3. clone DCASE repository
echo
echo "    3. Clone DCASE repository"
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

# 6. Create a symbolic link of our 'main_3.py' and 'core_3.py' from our repo 
# into 'audioset_classification'.
# then have a 'runme.sh' in our repo that will run the 'main_3.py' from
# 'audioset_classification
echo
echo "    6. Create a symbolik link of 'main_3.py' and 'core_3.py' in:"
echo "`pwd`" #/../audioset_classification/pytorch/"
sudo ln -s `pwd`/main_3.py `pwd`/audioset_classification/pytorch/
#echo "    main_3.py - linked"
sudo ln -s `pwd`/core_3.py `pwd`/audioset_classification/pytorch/
#echo "    core_3.py - linked"

#### remove all
#echo "\nRemove all..."
#echo `pwd`
#sh `pwd`/remove_all.sh # call another shell script

echo
echo
echo "***************"
echo "* Setup done! *"
echo "***************"
echo
