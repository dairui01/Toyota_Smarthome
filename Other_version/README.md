# Blanced TSU and Joint-view TSU

## Balanced TSU

Balanced TSU is a version of the dataset that overlooks the fine-grained details (e.g., the manipulated object) but keeps only the different movement patterns (e.g., cut, drink). There are many activity classes that have a limited number of instances (i.e., samples) in the fine-grained TSU version. This is because some activity classes with specific fine-grained details occur rarely as activity instances within the dataset which may not be sufficient to learn activity-specific representations. Thus to handle this, we release Balanced TSU, which focuses on the different movement patterns of the activity (i.e., verb) rather than the fine-grained details (i.e., noun). Balanced TSU shared the same untrimmed videos as fine-grained TSU: 536 videos with 21 minutes average duration. The only difference lies in the annotation. This version of the dataset merges the fine-grained activities that share similar motion into the same activity class. Therefore, this version of dataset is more balanced in terms of the number of samples, with slightly less number of classes (in total 34 activity classes).

![](/Images/Balanced_TSU.PNG)

## Joint-View TSU

The goal of the project is combine footage of same scenario taken by two different cameras placed in 2 diffrerent places and then perform action recognition. This repositry includes various experiments that were performed inconjunction with joint view.

<img src="/Images/Sit_down_v1.gif" width="300" height="200"/> <img src="/Images/Sit_down_v2.gif" width="300" height="200"/> 


### Introduction.

Joint view action detection/recognition is nothing but combining footage of same scenarios captured in different views to perform action detetction. This is a challenging task for any current state of the art action recognition methods because there is a huge difference in visual features across different views even though the scenario is same. The main aim of the project is to obtain a combined feature level representation/embedding from different views that can be used to improve action detection perdformance.This is done in two steps.

1). First we extract the spatio-embeddings of the videos using a pre-trained 3D-CNN network in our case its I3D[1].

2). Then a 2 stream TCN network is used to perform action detection wherein each stream is inputted with a embedding from a particular view and similarily with the other stream.

The main dataset that we used in this project is Toyota smarthome untrimmed dataset[2].

### Experiments and Terminology.
Along with joint view we have performed various experiments on the Toyota untrimmed smarthome dataset and have named each folder accordingly. The essential terms necessary to understand the project are .

* Balanced and Unbalanced Dataset.

The original Toyota untrimmed smarthome video dataset has close to 51 actions and the frequency of the occurence of this actions in the videos is highly unbalanced meaning certain actions occur more frequently than other actions and hence this is the reason we call it unbalanced datset. Later on in one of the [experiment](https://github.com/hari431996/joint_view_action_detection/tree/main/balanced_data_un_synchronised_footage%20) we removed certain low frequency actions and obtained a final balanced dataset consiting of only 34 highly occuring actions.

* Synchronised and unsynchronised.

Given a pair of videos of  two different views  of same scenario, should have overlapping footage or in other words they should have starting and end points in which the scenario is same but from different views, this is what we mean by synchronised footage. In the original Toyota untrimmed smarthome dataset there are close to 536 videos and a given pair of videos have non overlapping starting points but whereas in a synchronised pair there are only 298 videos and given pair of videos have starting and ending points with overlapping footage. Hence, the major difference between synchronised and unsynchronised is that the last one has more runtime than compared to synchronised videos.

* Joint View.

In joint view we utilise features from video pairs of two different views. There are close to 298 videos thereby giving us a pair of 149 videos. The features from this pairs of videos with different views are combined at various stages of a 2-Stream TCN network like for example late [fusion](https://github.com/hari431996/joint_view_action_detection/tree/main/joint_view_late_fusion).

### Datasets Overview.

There are three kinds of datasets used, they are present in each folder corresponding to the experiment they were utilised in.

1). smarthome_CS_51.json - This is original Toyota smarthome untrimmed dataset. The json file contains annotations of 536 videos and there are 51 class label actions.

2). unbalanced_data_synchronised_footage.json - This datset consists of annotations of 298 videos but the videos are in pairs and each pair have synchronised footage. Also the dataset contains, 51 label actions.

3). unsynchronised_balance_data_annotation.json - This dataset consists of annotations of 536 videos but the action labels are only 34 instead of 51 like in the previous datasets.


### References.

[1]. Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset, CVPR 2017

<a href="https://arxiv.org/abs/2010.14982" target="_blank">[2]. Toyota Smarthome Untrimmed: Real-World Untrimmed Videos for Activity Detection.</a>




