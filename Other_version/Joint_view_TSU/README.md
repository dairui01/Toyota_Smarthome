# Joint-view TSU

The `videos_sync.csv` provides the annotation to obtain the synchronized TSU from two views. 
After having the synchronized videos and extracted the I3D features, we can perform the join-view action detection. 

In this experiment, we use AGNet (the proposed baseline method in TSU) to fuse the information from the two synchronized views. The dataset used is unbalanced with 51 actions and the footage is synchronised.

1). models_xxx - This file basically contains network architecture/backbone.

2). smarthome_i3d_per_xxx - This file contains the dataloader necessary to fit the model.

3). train_xx - This file contains the code to train the model.

4). run_xx.sh - This file is a shell script that is necessary to run the model. Note - Please read carefully about each argument before running the experimen 

5). json file is the corresponding dataset annotation.

