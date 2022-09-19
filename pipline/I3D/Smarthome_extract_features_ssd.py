import os
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
import sys
import argparse
from multiprocessing import cpu_count

parser = argparse.ArgumentParser()
parser.add_argument('-mode', type=str, help='rgb or flow')
parser.add_argument('-load_model', type=str)
parser.add_argument('-root', type=str)
parser.add_argument('-gpu', type=str)
parser.add_argument('-save_dir', type=str)
parser.add_argument('-split', type=str)
parser.add_argument('-window_size', type=int)
args = parser.parse_args()
#print os.environ
#os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
#print os.environ["CUDA_VISIBLE_DEVICES"]

import torch
import torch._utils
try:
    torch._utils._rebuild_tensor_v2
except AttributeError:
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        tensor._backward_hooks = backward_hooks
        return tensor
    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable

import torchvision
from torchvision import datasets, transforms
import videotransforms


import numpy as np

from pytorch_i3d import InceptionI3d

from Smarthome_extract_features_ssd import TSU as Dataset

from tqdm import tqdm

import cv2

import os
import os.path


def video_to_tensor(pic):
    """Convert a ``numpy.ndarray`` to tensor.
    Converts a numpy.ndarray (T x H x W x C)
    to a torch.FloatTensor of shape (C x T x H x W)

    Args:
         pic (numpy.ndarray): Video to be converted to tensor.
    Returns:
         Tensor: Converted video.
    """
    # return torch.from_numpy(pic.transpose([3, 0, 1, 2]))
    return torch.from_numpy(pic.transpose([0, 4, 1, 2, 3]))


def load_rgb_frames(image_dir, vid, start, end):
    frames = []
    for i in tqdm(range(start, end)):
        img = cv2.imread(os.path.join(image_dir, vid, str(i).zfill(8) + '.jpg'))[:, :, [2, 1, 0]]
        w, h, c = img.shape
        if w < 224 or h < 224:
            d = 226. - min(w, h)
            sc = 1 + d / min(w, h)
            img = cv2.resize(img, dsize=(0, 0), fx=sc, fy=sc)
        img = (img / 255.) * 2 - 1
        frames.append(img)
    # print len(frames)
    while len(frames)<int(args.window_size):
        # print ('padding')
        frames.append(img)
    frames = np.asarray(frames, dtype=np.float32)
    return frames


def make_dataset(vid_name, root):
    dataset = []
    num_frames = len(os.listdir(os.path.join(root, vid_name)))
    dataset.append((vid_name, num_frames))
    return vid_name, num_frames


def prepare_data(save_dir,vid,mode,nf,root,start,end):
    if os.path.exists(os.path.join(save_dir, vid + '.npy')):
        return 0, 0, vid

    if mode == 'rgb':
        # shift_list = [t for t in np.linspace(1, int(nf) + 1, )]
        # images = []
        imgs_tem = load_rgb_frames(root, vid, start, end)
        imgs_tem = imgs_tem[np.newaxis, :, :, :, :]

    return video_to_tensor(imgs_tem), vid


def run(max_steps=64e3, mode='rgb', root='', vid_name='', batch_size=1, load_model='', save_dir=''):

    print root

    num_frames = len(os.listdir(os.path.join(root, vid_name)))
    print num_frames
    if os.path.exists(os.path.join(save_dir, vid_name + '.npy')):
        print split, 'feature exist!'
        exit()

    # setup the model
    if mode == 'flow':
        i3d = InceptionI3d(400, in_channels=2)
    else:
        i3d = InceptionI3d(48, in_channels=3) 

    i3d.load_state_dict(torch.load(load_model))
    i3d_till5c = nn.Sequential(*list(i3d._modules.values())[:-3])
    i3d = i3d.cuda()
    i3d_parallel = torch.nn.DataParallel(i3d)

    for phase in ['Extraction']:
        i3d_parallel.train(False)  # Set model to evaluate mode

        # get the inputs
        name, nf = make_dataset(vid_name, root)

        print os.path.join(save_dir, name+'.npy')
        if os.path.exists(os.path.join(save_dir, name+'.npy')):
            continue

        # b,c,t,h,w = inputs.shape
        print '--------------'
        print nf
        window_size = int(args.window_size)  # 16
        print window_size
        if nf > window_size:
            features = []
            for start in tqdm(range(1, nf, window_size)):
                # print start
                end = min(nf, start + window_size)
                start = max(1, start)
                # print (start,end)
                inputs, name = prepare_data(save_dir, vid_name, mode, nf, root, start, end)
                # print inputs.size()
                ip = Variable(torch.from_numpy(inputs.numpy()[:, :, :]).cuda(), volatile=True)
                # print i3d_parallel.module.extract_features(ip).squeeze(0).permute(1, 2, 3, 0).data.cpu().numpy().shape
                features.append(i3d_parallel.module.extract_features(ip).squeeze(0).permute(1, 2, 3, 0).data.cpu().numpy())
                # print 'finished'
            np.save(os.path.join(save_dir, name), np.concatenate(features, axis=0))
            print ('save_path:', os.path.join(save_dir, name))
        else:
            inputs, name = prepare_data(save_dir, vid_name, mode, nf, root, 1, nf)
            inputs = Variable(inputs.cuda(), volatile=True)
            features = i3d_parallel.module.extract_features(inputs)
            np.save(os.path.join(save_dir, name), features.squeeze(0).permute(1,2,3,0).data.cpu().numpy())
            print ('save_path:',os.path.join(save_dir, name))
        print 'finished:', nf, name


if __name__ == '__main__':
    # need to add argparse
    print('cuda_avail',torch.cuda.is_available())
    run(mode=args.mode, root=args.root, load_model=args.load_model, save_dir=args.save_dir, vid_name=args.split)