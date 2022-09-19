# I3D Feature Extraction

## Overview

This Repository contains the code for the I3D feature extraction on TSU dataset.

The I3D model indicates the model introduced by "[Quo Vadis, Action Recognition? A New Model and the Kinetics
Dataset](https://arxiv.org/abs/1705.07750)" by Joao Carreira and Andrew Zisserman. This code is based on Deepmind's [Kinetics-I3D](https://github.com/deepmind/kinetics-i3d). Including PyTorch versions of their models.

## Note
This code was written for PyTorch 0.3. The relevant dependencies, such as `opencv`, `tqdm`, `numpy` are also needed. Modified the environment in `Smarthome_SSD_16.sh` before run the code. 

## Feature Extraction

[Smarthome_extract_features_ssd.py](Smarthome_extract_features_ssd.py) contains the code to load a pre-trained I3D model and extract the features and save the features as numpy arrays. The code is processed by video.
The `pre-trained I3D model` can be downloaded via this [link](https://repo-sam.inria.fr/smarthome/TSU_16frame_weights_iter64000.pt). The authentication can be obtained while you request the data via our [website](https://project.inria.fr/toyotasmarthome/). 

## Quick start

`./Smarthome_SSD_16.sh video_list.txt`


