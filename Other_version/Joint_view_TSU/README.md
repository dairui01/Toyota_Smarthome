# Joint-view TSU
In this experiment, we use a 2 stream TCN architecture, where after there are 5 stages of TCN blocks and after each block theres a fusion so that thers information flow inbetween both the 2 streams. The dataset used is unbalanced with 51 actions and the footage is synchronised.

1). models_xxx - This file basically contains network architecture/backbone.

2). smarthome_i3d_per_xxx - This file contains the dataloader necessary to fit the model.

3). train_xx - This file contains the code to train the model.

4). run_xx.sh - This file is a shell script that is necessary to run the model. Note - Please read carefully about each argument before running the experimen 

5). json file is the corresponding dataset annotation.

