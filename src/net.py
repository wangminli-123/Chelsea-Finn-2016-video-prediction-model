'''
Author: bg
Date: 2020-11-11 14:02:34
LastEditTime: 2020-11-11 15:01:49
LastEditors: bg
Description: 
FilePath: /Chelsea-Finn-2016-video-prediction-model/src/net.py
'''
import math
import torch
import numpy as np
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F

class ConvLSTMCell(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, forget_bias=1.0, device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        super(ConvLSTMCell, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding if padding else int((kernel_size - 1) / 2)
        self.f_bias = forget_bias
        self.device = device

        self.Wxi = nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whi = nn.Conv2d(self.out_channels, self.out_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxf = nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whf = nn.Conv2d(self.out_channels, self.out_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxc = nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whc = nn.Conv2d(self.out_channels, self.out_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxo = nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Who = nn.Conv2d(self.out_channels, self.out_channels, self.kernel_size, 1, self.padding, bias=False)

    def forward(self, x, states):
        if states is None:
            states = self.stateInit(x)
        if not isinstance(states, tuple):
            raise TypeError("states type is not right")
        c, h = states
        ci = torch.sigmoid(self.Wxi(x) + self.Whi(h))
        cf = torch.sigmoid(self.Wxf(x) + self.Whf(h))
        cc = cf * c + ci * torch.tanh(self.Wxc(x) + self.Whc(h))
        co = torch.sigmoid(self.Wxo(x) + self.Who(h))
        ch = co * torch.tanh(cc)
        return ch, (cc, ch)

    def stateInit(self, x, output_shape=[0, 0]):
        shape = output_shape if output_shape[0] else [x.shape[2], x.shape[3]]
        states = (
            Variable(torch.zeros([x.shape[0], self.out_channels, shape[0], shape[1]])).to(self.device),
            Variable(torch.zeros([x.shape[0], self.out_channels, shape[0], shape[1]])).to(self.device)
            )
        return states

class PredNet(nn.Module):
    def __init__(
        self, img_shape=[3, 64, 64], # [channels, height, weight]
        lstm_size=[32, 32, 64, 64, 128, 64, 32],
        num_masks=10,
        is_self_mask=1,
        is_robot_state_used=1,
        context_frames=2,
        iter_num=-1,
        k=900,
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        ):
       
        super(PredNet, self).__init__()
        self.channels = img_shape[0]
        self.height = img_shape[1]
        self.width = img_shape[2]
        self.lstm_size = lstm_size
        self.num_masks = num_masks
        self.is_self_mask = is_self_mask 
        self.is_robot_state_used = is_robot_state_used # 是否使用机械臂的状态：action & states
        self.context_frames = context_frames
        self.CDNA_KERNEL_SIZE = 5
        self.iter_num = iter_num
        self.k = k
        self.device = device

        if not self.is_robot_state_used:
            self.POS_DIM = 0
            self.ACTION_DIM = 0
        else:
            self.POS_DIM = 3
            self.ACTION_DIM = 4
        
        self.layerInit()

    def layerInit(self):
        '''
        description: Define each layer of the model.
        '''
        self.conv1 = nn.Conv2d(in_channels=self.channels, out_channels=self.lstm_size[0], kernel_size=5, stride=2, padding=2)
        self.conv1_norm = nn.LayerNorm([self.lstm_size[0], self.height//2, self.width//2])
    
        self.lstm1 = ConvLSTMCell(in_channels=self.lstm_size[0], out_channels=self.lstm_size[0], kernel_size=5, stride = 1, padding=2, device = self.device)
        self.lstm1_norm = nn.LayerNorm([self.lstm_size[0], self.height//2, self.width//2])
    
        
        self.lstm2 = ConvLSTMCell(in_channels=self.lstm_size[0], out_channels=self.lstm_size[1], kernel_size=5, stride = 1, padding=2, device = self.device)
        self.lstm2_norm = nn.LayerNorm([self.lstm_size[1], self.height//2, self.width//2])
        
        self.stride1 = nn.Conv2d(in_channels=self.lstm_size[1], out_channels=self.lstm_size[1], kernel_size=3, stride=2, padding=1)
        
        self.lstm3 = ConvLSTMCell(in_channels=self.lstm_size[1], out_channels=self.lstm_size[2], kernel_size=5, stride = 1, padding=2, device = self.device)
        self.lstm3_norm = nn.LayerNorm([self.lstm_size[2], self.height//4, self.width//4])
            
        
        self.lstm4 = ConvLSTMCell(in_channels=self.lstm_size[2], out_channels=self.lstm_size[3], kernel_size=5, stride = 1, padding=2, device = self.device)
        self.lstm4_norm = nn.LayerNorm([self.lstm_size[3], self.height//4, self.width//4])

        self.stride2 = nn.Conv2d(in_channels=self.lstm_size[3], out_channels=self.lstm_size[3], kernel_size=3, stride=2, padding=1)
        self.robotConcat = nn.Conv2d(in_channels=self.lstm_size[3]+self.POS_DIM+self.ACTION_DIM, out_channels=self.lstm_size[3], kernel_size=1, stride=1)

        self.lstm5 = ConvLSTMCell(in_channels=self.lstm_size[3], out_channels=self.lstm_size[4], kernel_size=5, stride = 1, padding=2, device = self.device)
        self.lstm5_norm = nn.LayerNorm([self.lstm_size[4], self.height//8, self.width//8])

        self.deconv1 = nn.ConvTranspose2d(in_channels=self.lstm_size[4], out_channels=self.lstm_size[4], kernel_size=3, stride=2, output_padding=1, padding=1)
        
        self.lstm6 = ConvLSTMCell(in_channels=self.lstm_size[4], out_channels=self.lstm_size[5], kernel_size=5, stride = 1, padding=2, device = self.device)
        self.lstm6_norm = nn.LayerNorm([self.lstm_size[5], self.height//4, self.width//4])
        
        self.deconv2 = nn.ConvTranspose2d(in_channels=self.lstm_size[5] + self.lstm_size[1], out_channels=self.lstm_size[5] + self.lstm_size[1], kernel_size=3, stride=2, output_padding=1, padding=1)           
        
        self.lstm7 = ConvLSTMCell(in_channels=self.lstm_size[5] + self.lstm_size[1], out_channels=self.lstm_size[6], kernel_size=5, stride = 1, padding=2, device = self.device)
        self.lstm7_norm = nn.LayerNorm([self.lstm_size[6], self.height//2, self.width//2])
        
        self.deconv3 = nn.ConvTranspose2d(in_channels=self.lstm_size[6]+self.lstm_size[0], out_channels=self.lstm_size[6], kernel_size=3, stride=2, output_padding=1, padding=1)
        self.deconv3_norm = nn.LayerNorm([self.lstm_size[6], self.height, self.width])

        self.fc = nn.Linear(int(self.lstm_size[4] * self.height * self.width / 64), self.CDNA_KERNEL_SIZE * self.CDNA_KERNEL_SIZE * self.num_masks)
        self.maskout = nn.ConvTranspose2d(self.lstm_size[6], self.num_masks+1, kernel_size=1, stride=1)
        self.stateout = nn.Linear(self.POS_DIM+self.ACTION_DIM, self.POS_DIM)
        self.deconv4 = nn.ConvTranspose2d(in_channels=self.lstm_size[6], out_channels=self.channels, kernel_size=1, stride=1)
          
    def forward(self, images, actions, init_pos, train_process=True):
        '''
        :param inputs: T * N * C * H * W
        :param state: T * N  * C
        :param action: T * N * C
        :return:
        '''

        self.iter_num += 1 
        lstm_state1, lstm_state2, lstm_state3, lstm_state4, lstm_state5, lstm_state6, lstm_state7= None, None, None, None, None, None, None
        gen_images, gen_states = [], []
        init_pos = init_pos
        num_gt = int(np.round(images[0].shape[0] * (self.k / (math.exp(self.iter_num/self.k) + self.k))))
        
        for image, action in zip(images[:-1], actions[:-1]): 
            if len(gen_images) >= self.context_frames: # 预测到第三帧了
                if train_process:
                    image = self.scheduled_sample(image, gen_images[-1], num_gt)
                else:
                    image = gen_images[-1]
            else:
                image = image # 第三帧以前都是gt
            
            conv1 = self.conv1(image) # (b,3,64,64)->(b,32,32,32)
            conv1_norm = self.conv1_norm(conv1)

            lstm1, lstm_state1 = self.lstm1(conv1_norm, lstm_state1) # (b,3,32,32)->(b,32,32,32)
            lstm1_norm = self.lstm1_norm(lstm1)

            lstm2, lstm_state2 = self.lstm2(lstm1_norm, lstm_state2) # (b,32,32,32)->(b,32,32,32)
            lstm2_norm = self.lstm2_norm(lstm2)

            stride1 = self.stride1(lstm2_norm) # (b,32,32,32)->(b,32,16,16)

            lstm3, lstm_state3= self.lstm3(stride1, lstm_state3) # (b,32,16,16)->(b,64,16,16)
            lstm3_norm = self.lstm3_norm(lstm3)

            lstm4, lstm_state4= self.lstm4(lstm3_norm, lstm_state4) # (b,64,16,16)->(b,64,16,16)
            lstm4_norm = self.lstm4_norm(lstm4)

            stride2 = self.stride2(lstm4_norm) # (b,64,16,16)->(b,64,8,8)

            if self.is_robot_state_used:
                state_action = torch.cat([action, init_pos], dim=1)
                smear = torch.reshape(state_action, list(state_action.shape)+[1, 1])
                smear = smear.repeat(1, 1, 8, 8)
                stride2 = torch.cat([stride2, smear], dim=1) 
                
            robotconcat = self.robotConcat(stride2) # (b,64,8,8)

            lstm5, lstm_state5 = self.lstm5(robotconcat, lstm_state5) # (b,64,8,8)->(b,128,8,8)
            lstm5_norm = self.lstm5_norm(lstm5)

            deconv1 = self.deconv1(lstm5_norm) # (b,128,8,8)->(b,128,16,16)

            lstm6, lstm_state6= self.lstm6(deconv1, lstm_state6) # (b,128,16,16)->(b,64,16,16)
            lstm6_norm = self.lstm6_norm(lstm6)

            # skip connection
            lstm6_norm = torch.cat([lstm6_norm, torch.relu(stride1)], dim=1) # (b,64+32,16,16)
            deconv2 = self.deconv2(lstm6_norm) # (b,64+32,16,16)->(b,64+32,32,32)

            lstm7, lstm_state7 = self.lstm7(deconv2, lstm_state7) # (b,64+32,32,32)->(b,32,32,32)
            lstm7_norm = self.lstm7_norm(lstm7)
            # skip connection
            lstm7_norm = torch.cat([lstm7_norm, conv1_norm], dim=1) # (b,32+32,32,32)

            deconv3 = self.deconv3(lstm7_norm) # (b,32+32,32,32)->(b,32,64,64)
            deconv3_norm = self.deconv3_norm(deconv3)

            deconv4 = self.deconv4(torch.relu(deconv3_norm))
            transformed = [torch.sigmoid(deconv4)]
            input = lstm5_norm.view(lstm5_norm.shape[0], -1)

            transformed += self.cdna_transformation(image, input)

            maskout = self.maskout(torch.relu(deconv3_norm)) # (b,32,64,64)->(b,10+1,64,64)
            masks = torch.softmax(maskout, dim=1)
            mask_list = torch.split(masks,split_size_or_sections=1, dim=1)
        
            output = mask_list[0] * image
            for layer, mask in zip(transformed, mask_list[1:]):
                output += layer * mask

            current_state = self.stateout(state_action)
            gen_images.append(output)
            gen_states.append(current_state)

        return gen_images, gen_states
            
    
    def robot_process(robot_actions, robot_pos, shape = [8, 8]):
        '''
        description: Concat the robot_action vector and the robot_pos vector.

        param {robot_actions:list(shape=[batchbatch, 4]), robot_pos:list(shape=[batch, 3]), shape = [8, 8]} 
        
        return {torch.tensor in shape of [batch_size, dim(robot_action)+dim(robot_pos), shape]} 
        '''        
        state_action = torch.cat([robot_actions, robot_pos], dim=1)
        smear = torch.reshape(state_action, list(state_action.shape)+[1, 1])
        smear = smear.repeat(1, 1, shape[0], shape[1])
        return smear

    def cdna_transformation(self, image, cdna_input):
        '''
        description: 
        param {type} 
        return {type} 
        '''
        batch_size, height, width = image.shape[0], image.shape[2], image.shape[3]
        cdna_kerns = self.fc(cdna_input)
        cdna_kerns = cdna_kerns.view(batch_size, 10, 1, 5, 5)
        cdna_kerns = torch.relu(cdna_kerns - 1e-12) + 1e-12
        norm_factor = torch.sum(cdna_kerns, dim=[2,3,4], keepdim=True)
        cdna_kerns /= norm_factor

        cdna_kerns = cdna_kerns.view(batch_size*10, 1, 5, 5)
        image = image.permute([1, 0, 2, 3])

        transformed = torch.conv2d(image, cdna_kerns, stride=1, padding=[2, 2], groups=batch_size)

        transformed = transformed.view(3, batch_size, 10, height, width)
        transformed = transformed.permute([1, 0, 3, 4, 2])
        transformed = torch.unbind(transformed, dim=-1)
        return transformed

    def scheduled_sample(self, ground_truth_x, generated_x, num_ground_truth):
        generated_examps = torch.cat([ground_truth_x[:num_ground_truth], generated_x[num_ground_truth:]], dim=0)
        return generated_examps
