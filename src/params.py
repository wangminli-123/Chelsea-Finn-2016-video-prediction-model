'''
Author: your name
Date: 2020-10-16 15:22:58
LastEditTime: 2020-10-16 22:47:50
LastEditors: Please set LastEditors
Description: The model global parameters and system parameters
FilePath: /zcl_ws/src/C.FinnByLin/src/params.py
'''
import os
import argparse
import torch

class Params():
    '''
    description: The model global parameters and system parameters class.
    return [argparse]: argparse containing train and test options 
    '''
    def __init__(self, device):

        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        self.parser.add_argument('--raw_dataset_path', default='/home/wml/zcl_ws/src/C.FinnByLin/data/softmotion30_44k/', help='directory containing raw data.')
        self.parser.add_argument('--processed_path', default='/home/wml/zcl_ws/src/C.FinnByLin/data/processed/', help='directory containing raw data.')
        self.parser.add_argument('--output_dir', default='weight', help='directory for model weight.')

        self.parser.add_argument('--channels', type=int, default=3, help='# channel of input')
        self.parser.add_argument('--height', type=int, default=64, help='height of image')
        self.parser.add_argument('--width', type=int, default=64, help='width of image')

        # self.parser.add_argument('--pretrained_model', default='', help='filepath of a pretrained model to initialize from.')
        self.parser.add_argument('--sequence_length', type=int, default=10, help='sequence length, including context frames.')
        self.parser.add_argument('--context_frames', type=int, default=2, help= '# of frames before predictions.')
        self.parser.add_argument('--use_state',  default=True, action='store_true', help='Whether or not to give the state+action to the model')
        self.parser.add_argument('--num_masks', type=int, default=10, help='number of masks, usually 1 for DNA, 10 for CDNA, STN.')
        self.parser.add_argument('--device', default='cpu', help='cuda:[d] | cpu')

        # training details
        self.parser.add_argument('--print_interval', type=int, default=10, help='# iterations to output loss')
        self.parser.add_argument('--schedsamp_k', type=float, default=900.0, help='The k hyperparameter for scheduled sampling, -1 for no scheduled sampling.')
        self.parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
        self.parser.add_argument('--learning_rate', type=float, default=0.001, help='the base learning rate of the generator')
        self.parser.add_argument('--epochs', type=int, default=10, help='# total training epoch')
        self.opt = None


