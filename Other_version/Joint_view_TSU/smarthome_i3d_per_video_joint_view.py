import torch
import torch.utils.data as data_utl
from torch.utils.data.dataloader import default_collate

import numpy as np
import json
import csv
#import h5py

import os
import os.path
from tqdm import tqdm


def video_to_tensor(pic):
    """Convert a ``numpy.ndarray`` to tensor.
    Converts a numpy.ndarray (T x H x W x C)
    to a torch.FloatTensor of shape (C x T x H x W)
    
    Args:
         pic (numpy.ndarray): Video to be converted to tensor.
    Returns:
         Tensor: Converted video.
    """
    return torch.from_numpy(pic.transpose([3,0,1,2]))


def make_dataset(split_file, split, root, num_classes=48):
    dataset = []
    with open(split_file, 'r') as f:
        data = json.load(f)

    i = 0
    prev = ""
    key = list(data.keys())
    length = len(key)
    for vid in range(0, length, 2):
       
        if data[key[vid]]['subset'] != split:
            continue

        if not os.path.exists(os.path.join(root, key[vid]+'.npy')):
            print(root+vid)
            continue
        fts = np.load(os.path.join(root, key[vid]+'.npy'))
        num_feat = fts.shape[0]
        label = np.zeros((num_feat,num_classes), np.float32)
        
  
        fps = float(num_feat/float(data[key[vid]]['duration']))
        # print fps
        # print data[vid]['duration']
        for ann in data[key[vid]]['actions']:
            for fr in range(0,num_feat,1):
                # print (fr,num_feat,fps)
                if fr/fps > ann[1] and fr/fps < ann[2]:
                    label[fr, ann[0]] = 1 # binary classification
        dataset.append((key[vid], key[vid+1], label,data[key[vid]]['duration'] ,data[key[vid+1]]['duration']))
       
            
        i += 1
    
    return dataset

# make_dataset('multithumos.json', 'training', '/ssd2/thumos/val_i3d_rgb')


class MultiThumos(data_utl.Dataset):

    def __init__(self, split_file, split, root, batch_size, classes):
        
        self.data = make_dataset(split_file, split, root, classes)
        self.split_file = split_file
        self.batch_size = batch_size
        self.root = root
        self.in_mem = {}

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        entry = self.data[index]
        
        if entry[0] in self.in_mem and entry[1] in self.in_mem :
            feat_1 = self.in_mem[entry[0]]
            feat_2 = self.in_mem[entry[1]]
        else:
            
            #feature 1 from video 1
            if not os.path.exists(os.path.join(self.root, entry[0]+'.npy')) and not os.path.exists(os.path.join(self.root, entry[1]+'.npy')):
                print(entry[0], entry[1])
                return
            feat_1 = np.load(os.path.join(self.root, entry[0]+'.npy'))
            feat_1 = feat_1.reshape((feat_1.shape[0],1,1,1024))
            #r = np.random.randint(0,10)
            #feat = feat[:,r].reshape((feat.shape[0],1,1,1024))
            feat_1 = feat_1.astype(np.float32)
            self.in_mem[entry[0]] = feat_1
            
            #feature 2 from video 2
            feat_2 = np.load(os.path.join(self.root, entry[1]+'.npy'))
           
            feat_2 = feat_2.reshape((feat_2.shape[0],1,1,1024))
            #r = np.random.randint(0,10)
            #feat = feat[:,r].reshape((feat.shape[0],1,1,1024))
            feat_2 = feat_2.astype(np.float32)
            self.in_mem[entry[1]] = feat_2
            
           
            
            
            
            
        label = entry[2]
        return feat_1, feat_2,label, [entry[0],entry[1],entry[3], entry[4]]

    def __len__(self):
        return len(self.data)


def mt_collate_fn(batch):
        "Pads data and puts it into a tensor of same dimensions"
        max_len = 0
        for b in batch:
            if b[0].shape[0] > max_len:
                max_len = b[0].shape[0]
    
        new_batch = []
        for b in batch:
            
            #f_1 amd f_2 corresponds to features of video_1 and video_2 respectively
    
                
                f_1 = np.zeros((max_len, b[0].shape[1], b[0].shape[2], b[0].shape[3]), np.float32)
                f_2 = np.zeros((max_len, b[0].shape[1], b[0].shape[2], b[0].shape[3]), np.float32)
                
               
                m = np.zeros((max_len), np.float32)
                l = np.zeros((max_len, b[2].shape[1]), np.float32)
                f_1[:b[0].shape[0]] = b[0]
                f_2[:b[0].shape[0]] = b[0]
                m[:b[0].shape[0]] = 1
                #print(l.shape, b[0].shape, b[1].shape, b[2].shape)
                l[:b[0].shape[0], :] = b[2]
                new_batch.append([video_to_tensor(f_1),video_to_tensor(f_2) ,torch.from_numpy(m), torch.from_numpy(l), b[3]])
    
        return default_collate(new_batch)
    
