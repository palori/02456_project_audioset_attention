# SETUP FOLDERS TO RUN THE CODE #

cd temp_tests
# 1. download 'packed_features'
#https://drive.google.com/file/d/0B49XSFgf-0yVQk01eG92RHg4WTA/view?fbclid=IwAR2jhbhIbKbjA-cVdk_lMeEHTt__yqjo8su3yjjFXlPfu5RisZ96L51ld8E
# Permission denied!
wget "https://drive.google.com/uc?export=download&id=0B49XSFgf-0yVQk01eG92RHg4WTA"

# 2. clone AudioSet repository from https://github.com/qiuqiangkong/audioset_classification
#git clone https://github.com/qiuqiangkong/audioset_classification.git

# 3. clone DCASE repository
#git clone https://github.com/DCASE-REPO/dcase2018_baseline.git

# 4. generate data files
#python data_generator.py