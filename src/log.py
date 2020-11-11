'''
Author: bg
Date: 2020-11-11 14:02:34
LastEditTime: 2020-11-11 15:02:35
LastEditors: bg
Description: 
FilePath: /Chelsea-Finn-2016-video-prediction-model/src/log.py
'''
import time
import re

log_dir = '/home/wml/zcl_ws/src/C.FinnByLin/logs/'
log_time = time.asctime(time.localtime(time.time()))
log_time_list = re.findall('\w+', log_time)
t = log_time_list[1] + '_' + log_time_list[2] + '_' + log_time_list[3] + '_' + log_time_list[4]
log_name = log_dir + t + '_logfile.txt'

# def write_log_line(loss, ):

