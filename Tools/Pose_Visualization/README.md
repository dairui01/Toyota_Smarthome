# 2D Pose Visualizaiton

![](./test.jpg)


In this repo, we provide a simple visualization code for 2D Pose on our dataset. 
- The `json_to_npz.py` can transfer the json to the npz style data. 
- `skeleton_visualization_single_image.py` is used to visualize the 2D pose on a single image. 
- `skeleton_visualization_multi_image.py` is used to visualize the 2D pose on multiple images. 

## Quick start

1. `python json_to_npz.py json_name output_npz`
2. Change the image and skeleton file in `skeleton_visualization_single_image.py` and run it. The output test.jpg is the composition of 2D Pose and rgb image. 

# 3D Pose visualization

All the data is in world space,

- Joint list is like this ['Rank', 'Lank', 'Rkne', 'Lkne', 'Rhip', 'Lhip', 'Rwri', 'Lwri', 'Relb', 'Lelb', 'Rsho', 'Lsho', 'head'],

- Using 'Rsho' and 'Lsho' to get the neck position,

- Using 'Rhip' and 'Lhip' to get center hip position,

- Using 'Rsho', 'Lsho', 'Rhip' and 'Lhip' to get chest position,

- Nose position is overlapping with head position.

Also the translation data order is [X, X, X, …, Y, Y, Y, …., Z, Z, Z].

 


