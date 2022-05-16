# TSU pipline

This is the pipline code for Toyota Smarthome Untrimmed dataset (fine-grained version). The dataset description and original video can be request in this [project page](https://project.inria.fr/toyotasmarthome/).

## Prepare the I3D feature
Like the previous works for action detection, the model can be built on top of the pre-trained I3D features. Thus, feature extraction is needed before training the network.  We have preextracted the I3D feature for RGB and 3D pose feature for AGCN. Here are the download links:
- RGB-I3D feature [link]()
- 3D Pose-AGCN feature [link]()

## Dependencies 
Please satisfy the following dependencies to train MS-TCT correctly: 
- pytorch 1.9
- python 3.8 
- timm 0.4.12
- pickle5
- scikit-learn
- numpy

## Quick Start
1. Change the _rgb_root_ or _skeleton_root_ to the extracted feature path in the _train.py_. 
2. Use `./run.sh` for training on TSU-RGB. Evaluate with skeleton can be realized by changing the _-mode_ in _run.sh_ to _skeleton_.
3. The method is evaluate by the per-frame mAP. The event-based mAP can be found at this [repo](https://github.com/dairui01/TSU_evaluation/tree/main/Event_map).

## Reference
If you find our repo or paper useful, please cite us as
```bibtex
@ARTICLE{Dai_2022_PAMI,
  author={Dai, Rui and Das, Srijan and Sharma, Saurav and Minciullo, Luca and Garattoni, Lorenzo and Bremond, Francois and Francesca, Gianpiero},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={Toyota Smarthome Untrimmed: Real-World Untrimmed Videos for Activity Detection}, 
  year={2022},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TPAMI.2022.3169976}}
```


