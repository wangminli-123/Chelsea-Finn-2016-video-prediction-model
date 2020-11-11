'''
Author: bg
Date: 2020-11-11 14:02:34
LastEditTime: 2020-11-11 15:00:21
LastEditors: bg
Description: prepare the data as dataset class.
FilePath: /Chelsea-Finn-2016-video-prediction-model/src/data.py
'''
from pickle import decode_long
import re
import os
from typing import Generator
import numpy as np
import cv2
from numpy.core.fromnumeric import repeat
import tfrecord
from tfrecord.iterator_utils import sample_iterators
import torch
import time
from torch.utils.data import dataset

from torch.utils.data.dataloader import DataLoader
from torch.utils.data.distributed import DistributedSampler

raw_dataset_path = '/home/wml/zcl_ws/src/C.FinnByLin/data/softmotion30_44k/'
processed_path = "/home/wml/zcl_ws/src/C.FinnByLin/data/processed/"
record_dir = ["test/", "train/"]
feature_list = ['/image_aux1/encoded', '/endeffector_pos', '/action']
batch_size = 32
# feature_list = ['/image_aux1/encoded', '/image_main/encoded', '/endeffector_pos', '/action']

def convert():
    '''
    description: convert data/softmotion30_44K/* from .tfrecords to .npy
    
    file format: [30 * [image_aux1:np.ndarry, endeffector_pos:np.ndarry, action:np.ndarry]]
    '''
    try:
        start = time.time() 
        for dir in record_dir:
            for file in os.listdir(raw_dataset_path + dir):
                print("Converting " + file + "...")
                test_loader = tfrecord.tfrecord_loader(raw_dataset_path + dir + file, None)
                sample_num = 0
                for recordfile in test_loader:
                    sample = []
                    write_path = processed_path + dir
                    # print(write_path)
                    if not os.path.exists(write_path):
                        os.makedirs(write_path)
                    for i in range(30):
                        img = cv2.cvtColor(recordfile[str(i) + feature_list[0]].reshape([64, 64, 3]), cv2.COLOR_RGB2BGR)
                        pos = recordfile[str(i) + feature_list[1]]
                        act = recordfile[str(i) + feature_list[2]]
                        sample.append([img, pos, act])
                        # break
                    samples = (np.array(sample[:10]), np.array(sample[10:20]), np.array(sample[20:]))
                    for i in range(3):
                        np.save(write_path + file + str(sample_num) + '_' + str(i) +'.npy', samples[i], allow_pickle=True, fix_imports=True)
                        # print(len(samples[i]))
                    sample_num += 1 
                    break
                break
            break
        end = time.time()
        print("All dataset is converted! Consume " + str(end-start) + "s")
    except:
        raise (RuntimeError("Error occurs in data.py: module convert()"))

def getFilePath(path): 
    file_path_list = []
    for i in os.listdir(path):
        file_path_list.append(path + i) 
    if len(file_path_list) == 0:
        raise (RuntimeError("Found 0 file in folder: " + path))
    return file_path_list
    
def loader(path):
    samples = np.load(path, allow_pickle=True)
    if not samples.shape == (10, 3):
        raise (RuntimeError(".npy file damaged"))
    return samples
    
class PushDataset(torch.utils.data.Dataset):
    def __init__(self, root, device, train = True, image_transform=None, action_transform=None, state_transform=None, loader=loader): 
        '''
        description: 
        root:'/home/wml/zcl_ws/src/C.FinnByLin/data/processed'
        '''
        self.image_transform = image_transform
        self.action_transform = action_transform
        self.state_transform = state_transform

        self.device = device
        
        self.train = train
        self.loader = loader
        self.get_item_num = 0

        if not os.path.exists(root):
            raise FileExistsError('{0} does not exists!'.format(root))
                
        if self.train:
            self.data_path = root + '/train/'
        else:
            self.data_path = root + '/test/'

        self.file_list = getFilePath(self.data_path)
        self.file_list.sort(key=lambda x : (int(re.findall('\d+', x)[0]), int(re.findall('\d+', x)[2]), int(re.findall('\d+', x)[3])), reverse=False)

    def __getitem__(self, index):

        item_file = self.file_list[index]
        samples = loader(item_file)
        img_list, pos_list, action_list = [], [], []
        for num, (i, p, a) in enumerate(samples, 0):
            img_list.append(torch.from_numpy(np.transpose(i, (2,0,1)))/(1.0))######################
            pos_list.append(torch.from_numpy(p))
            action_list.append(torch.from_numpy(a))
        img = torch.cat([i.unsqueeze(0) for i in img_list], dim=0).to(self.device)
        pos = torch.cat([i.unsqueeze(0) for i in pos_list], dim=0).to(self.device)
        action = torch.cat([i.unsqueeze(0) for i in action_list], dim=0).to(self.device)
        return img.to(self.device), action.to(self.device), pos.to(self.device)

    def __len__(self):
        return len(self.file_list)

def build_dataloader(device):
    train_ds = PushDataset(
        root='/home/wml/zcl_ws/src/C.FinnByLin/data/processed',
        loader=loader,
        device=device
    )
    test_ds = PushDataset(
        root='/home/wml/zcl_ws/src/C.FinnByLin/data/processed',
        train=False,
        loader=loader,
        device=device
    )
    train_dl = DataLoader(dataset=train_ds, batch_size = batch_size, shuffle=True, drop_last=False)
    test_dl = DataLoader(dataset=test_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    return train_dl, test_dl, train_ds.__len__(), test_ds.__len__()     