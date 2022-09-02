import json
import os
import numpy as np
import sys
# posemap
coco=['nose', 'Leye' ,'Reye' ,'Lear' ,'Rear' ,'Lsho' ,'Rsho', 'Lelb', 'Relb', 'Lwri', 'Rwri', 'Lhip', 'Rhip', 'Lkne', 'Rkne', 'Lank' ,'Rank']
lcrnet={'nose': 12, 'Leye':12 ,'Reye':12 ,'Lear':12 ,'Rear':12 ,'Lsho':11 ,'Rsho':10, 'Lelb':9, 'Relb':8, 'Lwri':7, 'Rwri':6, 'Lhip':5, 'Rhip':4, 'Lkne':3, 'Rkne':2, 'Lank':1 ,'Rank':0}
#op={'nose': 0, 'Leye':0 ,'Reye':2 ,'Lear':3 ,'Rear':4 ,'Lsho':5 ,'Rsho':6, 'Lelb':7, 'Relb':8, 'Lwri':9, 'Rwri':10, 'Lhip':11, 'Rhip':12, 'Lkne':13, 'Rkne':14, 'Lank':15 ,'Rank':16}

name=sys.argv[1]
name2=sys.argv[2]
with open(name) as json_data:
    pose_pre = json.load(json_data)

kpts = []
for i in range(len(pose_pre['frames'])):
    if pose_pre['frames'][i]==[]:
        kpt0=[0]*36

    else:
        kpt0 = pose_pre['frames'][i][0]['pose2d']
    kpt=[[kpt0[lcrnet[j]], kpt0[lcrnet[j]+13]] for j in coco]
    kpts.append(np.array(kpt))


print('Usage: python json2lcrnet.py <jsonfile> <outputfolder>')
kpts = np.array(kpts).astype(np.float32)
name2=name2+name.split('/')[-1][:-5]+'_LCRNet2d.npz'
print('kpts npz saved in ', name2)
np.savez_compressed(name2, kpts=kpts)