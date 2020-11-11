'''
Author: bg
Date: 2020-11-11 14:02:34
LastEditTime: 2020-11-11 15:02:23
LastEditors: bg
Description: 
FilePath: /Chelsea-Finn-2016-video-prediction-model/src/train.py
'''
from options import Options

opt = Options().parse()
model = Model(opt)

opt = Options()
opt.parse()
