# TSU pipeline

This is the pipeline code for Toyota Smarthome Untrimmed dataset (fine-grained version). The dataset description and original video can be request in this [project page](https://project.inria.fr/toyotasmarthome/).

## Prepare the I3D feature
Like the previous works for action detection, the model can be built on top of the pre-trained I3D features. Thus, feature extraction is needed before training the network (See folder `./I3D`). 

To evaluate the pipeline, here, we have pre-extracted the I3D feature for RGB and 3D pose feature for AGCN. Those features can be downloaded in [dataset website](https://project.inria.fr/toyotasmarthome/).

## Dependencies 
Please satisfy the following dependencies to use this code correctly: 
- pytorch 1.9
- python 3.8 
- timm 0.4.12
- pickle5
- scikit-learn
- numpy

## Model
In this repository, you can train and test with [PDAN (WACV2021)](https://openaccess.thecvf.com/content/WACV2021/html/Dai_PDAN_Pyramid_Dilated_Attention_Network_for_Action_Detection_WACV_2021_paper.html).  

A pre-trained model (PDAN_TSU_RGB) for RGB modality and CS protocol can be downloaded in this [link](https://mybox.inria.fr/f/5e006560efaf4e0fb7ac/). The obtained f-mAP should be around 32.7%. For loading the pretrained model, you can add/change the `-load_model` parameter to the pre-trained model path (e.g., `'./PDAN_TSU_RGB'`).

The training hyperparameters for MS-TCT (CVPR'22) on TSU can be found [here](https://github.com/dairui01/MS-TCT/issues/4).

## Quick Start
1. Change the _rgb_root_ or _skeleton_root_ to the extracted feature path in the _train.py_. 
2. Use `./run_PDAN.sh` for training on TSU-RGB. Evaluation with skeleton can be realized by changing the `-mode` in _run.sh_ to _skeleton_.
3. The method is evaluated by the per-frame mAP. The event-based mAP can be found at this [repo](https://github.com/dairui01/TSU_evaluation/tree/main/Event_map).
4. For diiferent split settings (e.g. Cross-Subject or Cross-View), you can change it to either _CS_ or _CV_ for the `-split_setting` parameter in the command. 


## Reference
If you find our repo or paper useful, please cite us as
```bibtex
@article{dai2022toyota,
  title={Toyota smarthome untrimmed: Real-world untrimmed videos for activity detection},
  author={Dai, Rui and Das, Srijan and Sharma, Saurav and Minciullo, Luca and Garattoni, Lorenzo and Bremond, Francois and Francesca, Gianpiero},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2022},
  publisher={IEEE}
}
  
@inproceedings{dai2021pdan,
  title={Pdan: Pyramid dilated attention network for action detection},
  author={Dai, Rui and Das, Srijan and Minciullo, Luca and Garattoni, Lorenzo and Francesca, Gianpiero and Bremond, Francois},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages={2970--2979},
  year={2021}
}

```


