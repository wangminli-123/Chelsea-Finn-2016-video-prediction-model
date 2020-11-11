'''
Author: bg
Date: 2020-11-11 14:02:34
LastEditTime: 2020-11-11 15:02:05
LastEditors: bg
Description: 
FilePath: /Chelsea-Finn-2016-video-prediction-model/src/model.py
'''
import time
from net import PredNet
from data import build_dataloader

import os 
import time
from typing import NewType
import re
import numpy as np
from numpy.core.fromnumeric import shape
from numpy.lib.type_check import imag
from progress.bar import *
import torch
from torch import nn
from torch._C import device
from torch.nn import functional as F
from net import PredNet
from data import build_dataloader
from torch.utils.tensorboard import SummaryWriter
import cv2
from options import *

class Model():
    def __init__(self, opt):
        '''
        opt in type <class 'argparse.Namespace'>
        '''
        self.opt = opt
        self.device = self.opt.device

        train_dataloader, test_dataloader, self.train_num, self.test_num= build_dataloader(self.device)
        self.dataloader = {'train': train_dataloader, 'test': test_dataloader}

        self.net = PredNet(
            img_shape=self.opt.shape,
            num_masks=self.opt.num_masks,
            is_robot_state_used=1,
            iter_num=-1,
            k=900,
            device = self.device
            )
        self.net.to(self.device) 

        print('Net has',sum(param.numel() for param in self.net.parameters()), 'parameters...')

        self.mse_loss = torch.nn.MSELoss()
        self.w_state = 1e-4 # TODO problems

        if self.opt.pretrained_model_path:
            self.load_weight()
            
        self.optimizer = torch.optim.Adam(self.net.parameters(), self.opt.learning_rate)

    def peak_signal_to_noise_ratio(self, true, pred):
        #TODO normalization
        return 10.0 * torch.log(torch.tensor(255.0*255) / F.mse_loss(true, pred)) / torch.log(torch.tensor(10.0))

    def loss_calculation(images, gen_images, states, gen_states):
        loss, psnr = 0.0, 0.0
        for i, (image, gen_image) in enumerate(zip(images[self.opt.context_frames:], gen_images[self.opt.context_frames-1:])):
            recon_loss = self.mse_loss(image, gen_image)
            psnr_i = self.peak_signal_to_noise_ratio(image, gen_image)
            loss += recon_loss
            psnr += psnr_i
        # TODO state loss fixing
        for i, (state, gen_state) in enumerate(zip(states[self.opt.context_frames:], gen_states[self.opt.context_frames-1:])):
            state_loss = self.mse_loss(state, gen_state) * self.w_state
            loss += state_loss
            
        loss /= torch.tensor(self.opt.sequence_length - self.opt.context_frames)
        psnr /= torch.tensor(self.opt.sequence_length - self.opt.context_frames)
        return loss, psnr

    def train_epoch(self, epoch):
        consuming_time = 0
        loss = 100000 #TODO problems
        with Bar('training epoch:%d ' % epoch, max=self.train_num/self.opt.batch_size) as bar:#TODO bar fix
            start = time.time()
            
            for iter_num, (images, actions, states) in enumerate(self.dataloader['train']):
                self.net.zero_grad()
                images = images.permute([1, 0, 2, 3, 4]).unbind(0)
                actions = actions.permute([1, 0, 2]).unbind(0)
                states = states.permute([1, 0, 2]).unbind(0)
                              
                # gen_images, gen_states = self.net(images, actions, states[0], train_process=False)
                gen_images, gen_states = self.net(images, actions, states[0])
                loss, psnr = loss_calculation(images, gen_images, states, gen_states)
                loss.backward()
                self.optimizer.step()

                now = time.time()
                consuming_time = (now-start)/60 #minutes
                bar.next()
                print(' consuming:%.2f mins, loss:%6f, psnr:%6f' % (consuming_time,loss,psnr))
                
            bar.finish()

    def evaluate(self, epoch):
        with torch.no_grad():
            recon_loss, state_loss, psnr = 0.0, 0.0, 0.0
            for iter_, (images, actions, states) in enumerate(self.dataloader['test']):
                images = images.permute([1, 0, 2, 3, 4]).unbind(0)
                actions = actions.permute([1, 0, 2]).unbind(0)
                states = states.permute([1, 0, 2]).unbind(0)
                gen_images, gen_states = self.net(images, actions, states[0],train_process=False)
                for i, (image, gen_image) in enumerate(
                        zip(images[self.opt.context_frames:], gen_images[self.opt.context_frames - 1:])):
                    recon_loss += self.mse_loss(image, gen_image)
                    psnr_i = peak_signal_to_noise_ratio(image, gen_image)
                    psnr += psnr_i

                for i, (state, gen_state) in enumerate(
                        zip(states[self.opt.context_frames:], gen_states[self.opt.context_frames - 1:])):
                    state_loss += self.mse_loss(state, gen_state) * self.w_state
            recon_loss /= (torch.tensor(self.opt.sequence_length - self.opt.context_frames) * len(self.dataloader['test'].dataset)//self.opt.batch_size)
            state_loss /= (torch.tensor(self.opt.sequence_length - self.opt.context_frames) * len(self.dataloader['test'].dataset)//self.opt.batch_size)
            psnr /= (torch.tensor(self.opt.sequence_length - self.opt.context_frames) * len(self.dataloader['test'].dataset)//self.opt.batch_size)
            print("evaluation epoch: %3d, recon_loss: %6f, state_loss: %6f" % (epoch, recon_loss, psnr))

    def save_weight(self, epoch):
        log_time = time.asctime(time.localtime(time.time()))
        log_time_list = re.findall('\w+', log_time)
        t = log_time_list[1] + '_' + log_time_list[2] + '_' + log_time_list[3] + '_' + log_time_list[4]
        torch.save(self.net.state_dict(), os.path.join(self.opt.output_dir, t + "net_epoch_%d.pth" % epoch))

    def load_weight(self):
        print("loading pre-trained model...")
        self.net.load_state_dict(torch.load(self.opt.pretrained_model))
        print("pretrained model loaded")

    def train(self):
        for epoch_i in range(0, self.opt.epochs):
            self.train_epoch(epoch_i)
            self.evaluate(epoch_i)
            self.save_weight(epoch_i)

    def evaluate_show(self):
        with torch.no_grad():
            write_net = 1
            for iter_num, (images, actions, states) in enumerate(self.dataloader['test']):
                self.net.zero_grad()
                images = images.permute([1, 0, 2, 3, 4]).unbind(0)
                actions = actions.permute([1, 0, 2]).unbind(0)
                states = states.permute([1, 0, 2]).unbind(0)               
                gen_images, gen_states = self.net(images, actions, states[0])
                for t, (raw, gene) in enumerate(zip(images[1:], gen_images[0:])):
                    file_dir = '/home/wml/zcl_ws/src/C.FinnByLin/evaluate/'+ str(t)
                    raw_img = raw[0].cpu().numpy()
                    raw_img = np.transpose(raw_img, (1,2,0))
                    cv2.imwrite(file_dir + 'raw.jpg', raw_img)
                    gene_img = gene[0].cpu().numpy()
                    gene_img = np.transpose(gene_img, (1,2,0))
                    cv2.imwrite(file_dir + 'gene.jpg', gene_img)
                break
            
if __name__ == '__main__':
    opt = Options().parse()
    model = Model(opt)
    model.train()

    

    

