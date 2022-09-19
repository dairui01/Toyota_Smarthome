#!/usr/bin/env bash

export PATH=/anaconda/env/pytoch0.3/

echo "start..."

cat $1 | while read line
do
echo $line

python Smarthome_extract_features_ssd.py -window_size 16 -gpu 4 -split $line -mode rgb -root /path/to/smarthome_SSD_blurred -load_model /data/stars/user/rdai/PhD_work/cvpr2020/pytorch_i3d_new/models/ADF_16frame_weights_iter64000.pt -save_dir /data/stars/user/rdai/smarthome_untrimmed/features/i3d_16frames_64000

echo 'finish one video!'
done


