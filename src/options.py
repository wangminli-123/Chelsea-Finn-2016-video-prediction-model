'''
Author: bg
Date: 2020-11-11 14:02:34
LastEditTime: 2020-11-11 15:01:39
LastEditors: bg
Description: 
FilePath: /Chelsea-Finn-2016-video-prediction-model/src/options.py
'''
import os
import argparse
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
ws_path = os.path.abspath('sys.py')[:-6]
data_path = ws_path + 'data/'
log_path = ws_path + 'log/'

class Options():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        self.parser.add_argument('--ws_dir', default=ws_path, help='workspace directory.')
        self.parser.add_argument('--channels', type=int, default=3, help='# channel of input')
        self.parser.add_argument('--height', type=int, default=64, help='height of image')
        self.parser.add_argument('--width', type=int, default=64, help='width of image')
        self.parser.add_argument('--results_dir', default=ws_path+'results/', help='directory saves model weight.')
        self.parser.add_argument('--pretrained_model_path', default='', help='filepath of a pretrained model to initialize from.')
        self.parser.add_argument('--sequence_length', type=int, default=10, help='sequence length, including context frames.')
        self.parser.add_argument('--context_frames', type=int, default=2, help= '# of frames before predictions.')
        self.parser.add_argument('--use_state',  default=True, action='store_true', help='Whether or not to give the state+action to the model')
        self.parser.add_argument('--num_masks', type=int, default=10, help='number of masks, default 10')
        self.parser.add_argument('--device', default=device, help='cuda:[d] | cpu')

        # training details
        self.parser.add_argument('--schedsamp_k', type=float, default=900.0, help='The k hyperparameter for scheduled sampling, -1 for no scheduled sampling.')
        self.parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
        self.parser.add_argument('--learning_rate', type=float, default=0.001, help='the base learning rate of the generator')
        self.parser.add_argument('--epochs', type=int, default=10, help='# total training epoch')
        self.opt = None

    def parse(self):
        """ Parse Arguments.
        """
        self.opt = self.parser.parse_args()
        if not os.path.exists(self.opt.results_dir):
            os.makedirs(self.opt.results_dir)
        return self.opt
