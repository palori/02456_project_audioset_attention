#!/bin/bash


# RIGHT NOW IT'S JUST A COPY FROM THE AUDIOSET CLASSIFICATION, MODIFY TO RUN OUR CODE!


# You need to modify the dataset path.
# "/vol/vssp/msos/audioset/packed_features"

DATA_DIR="`pwd`/data"
#DATA_DIR="`pwd`/../packed_features"

# You can to modify to your own workspace. 
# WORKSPACE=`pwd`
# "/vol/vssp/msos/qk/workspaces/pub_audioset_classification"
WORKSPACE="../audioset_results_temp"

BACKEND="../audioset_classification/pytorch"     # 'pytorch' | 'keras'

MODEL_TYPE="decision_level_single_attention"    # 'decision_level_max_pooling'
                                                # | 'decision_level_average_pooling'
                                                # | 'decision_level_single_attention'
                                                # | 'decision_level_multi_attention'

# Train
CUDA_VISIBLE_DEVICES=1 python $BACKEND/main_3.py --data_dir=$DATA_DIR --workspace=$WORKSPACE --model_type=$MODEL_TYPE --mini_data train


# Calculate averaged statistics. 
python $BACKEND/main_3.py --data_dir=$DATA_DIR --workspace=$WORKSPACE --model_type=$MODEL_TYPE --mini_data get_avg_stats

