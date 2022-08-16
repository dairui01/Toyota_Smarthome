# Blanced TSU and Joint-view TSU

## Balanced TSU

Balanced TSU is a version of the dataset that overlooks the fine-grained details (e.g., the manipulated object) but keeps only the different movement patterns (e.g., cut, drink). There are many activity classes that have a limited number of instances (i.e., samples) in the fine-grained TSU version. This is because some activity classes with specific fine-grained details occur rarely as activity instances within the dataset which may not be sufficient to learn activity-specific representations. Thus to handle this, we release Balanced TSU, which focuses on the different movement patterns of the activity (i.e., verb) rather than the fine-grained details (i.e., noun). Balanced TSU shared the same untrimmed videos as fine-grained TSU: 536 videos with 21 minutes average duration. The only difference lies in the annotation. This version of the dataset merges the fine-grained activities that share similar motion into the same activity class. Therefore, this version of dataset is more balanced in terms of the number of samples, with slightly less number of classes (in total 34 activity classes).

![](./Balanced_TSU.PNG)

## Joint-View TSU

Joint-view TSU targets the joint-view activity detection task. Different from the aforementioned versions, this version of dataset contains only synchronized video pairs to be used for learning joint-view activity detection models.

The goal of the project is to combine footage of the same scenario taken by two different cameras placed in 2 different locations and then perform action detection. This repository includes various experiments that were performed in conjunction with the joint view.

<img src="./Sit_down_v1.gif" width="300" height="200"/> <img src="./Sit_down_v2.gif" width="300" height="200"/> 


### Introduction.

Joint-view action detection is to combine footage of the same scenarios captured in different views to perform action detection. This is a challenging task for any current state-of-the-art action recognition method because there is a huge difference in visual features across different views even though the scenario is the same. The main aim of the project is to obtain a combined feature-level representation/embedding from different views that can be used to improve action detection performance. This is done in two steps.

1). First we extract the spatio-embeddings of the videos using a pre-trained 3D-CNN network in our case its I3D[1].

2). Then a 2 stream TCN network is used to perform action detection wherein each stream is inputted with an embedding from a particular view and similarily to the other stream.

The main dataset that we used in this project is Toyota Smarthome untrimmed dataset[2].

## Experiments and Terminology.
Along with joint view, we have performed various experiments on the Toyota Smarthome untrimmed dataset and have named each folder accordingly. The essential terms necessary to understand the project are:

* Balanced and Unbalanced Dataset.

The original Toyota untrimmed smarthome video dataset has close to 51 actions and the frequency of the occurrence of these actions in the videos is highly unbalanced meaning certain actions occur more frequently than other actions and hence this is the reason we call it an unbalanced dataset. Later on, in one of the experiments we removed certain low-frequency actions and obtained a final balanced dataset consisting of only 34 highly occurring actions.

* Synchronised and unsynchronised.

Given a pair of videos of  two different views  of same scenario, should have overlapping footage or in other words they should have starting and end points in which the scenario is same but from different views, this is what we mean by synchronised footage. In the original Toyota untrimmed smarthome dataset there are close to 536 videos and a given pair of videos have non overlapping starting points but whereas in a synchronised pair there are only 298 videos and given pair of videos have starting and ending points with overlapping footage. Hence, the major difference between synchronised and unsynchronised is that the last one has more runtime than compared to synchronised videos.

* Joint View.

In joint view we utilise features from video pairs of two different views. There are close to 298 videos thereby giving us a pair of 149 videos. The features from these pairs of videos with different views are combined at various stages of a 2-Stream TCN network like for example late [fusion](https://github.com/hari431996/joint_view_action_detection/tree/main/joint_view_late_fusion).

## JSON Overview.

There are three kinds of datasets used, they are present in each folder corresponding to the experiment they were utilised in.

1). smarthome_CS_51.json - This is the original Toyota Smarthome untrimmed dataset (CS protocol). The json file contains annotations of 536 videos and there are 51 class label actions.

2). unbalanced_data_synchronised_footage.json - This dataset consists of annotations of 298 videos but the videos are in pairs and each pair have synchronised footage. Also the dataset contains 51 label actions.

3). unsynchronised_balance_data_annotation.json - This dataset consists of annotations of 536 videos but the action labels are only 34 instead of 51 like in the previous datasets.


### References.

[1]. Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset, CVPR 2017

<a href="https://arxiv.org/abs/2010.14982" target="_blank">[2]. Toyota Smarthome Untrimmed: Real-World Untrimmed Videos for Activity Detection.</a>
