###############----AGNet-Modified--------#################

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import os
import copy
import sys

class AGNet(nn.Module):
    def __init__(self, out_channels_y=512, num_f_maps=512, dim=1024, num_classes=157):
        super(AGNet, self).__init__()
        self.in_channels= num_f_maps
        self.out_channels = num_f_maps
        self.dim = dim
        self.num_classes = num_classes
        self.out_channels_y = out_channels_y
        self.conv1_x = nn.Conv1d(self.in_channels, self.out_channels, 3, padding=1, dilation=1)
        self.conv2_x = nn.Conv1d(self.in_channels, self.out_channels, 3, padding=2, dilation=2)
        self.conv3_x = nn.Conv1d(self.in_channels, self.out_channels, 3, padding=4, dilation=4)
        self.conv4_x = nn.Conv1d(self.in_channels, self.out_channels, 3, padding=8, dilation=8)
        self.conv5_x = nn.Conv1d(self.in_channels, self.out_channels, 3, padding=16, dilation=16)
        self.Drop1 = nn.Dropout()
        self.Drop2 = nn.Dropout()
        self.Drop3 = nn.Dropout()
        self.Drop4 = nn.Dropout()
        self.Drop5 = nn.Dropout()
        self.bottle0_x = nn.Conv1d(self.dim, self.out_channels, 1)
        self.bottle0_y = nn.Conv1d(self.dim, self.out_channels_y, 1)

        self.conv1_y = nn.Conv1d(self.out_channels_y, self.out_channels_y, 3, padding=1, dilation=1)
        self.conv2_y = nn.Conv1d(self.out_channels_y, self.out_channels_y, 3, padding=2, dilation=2)
        self.conv3_y = nn.Conv1d(self.out_channels_y, self.out_channels_y, 3, padding=4, dilation=4)
        self.conv4_y = nn.Conv1d(self.out_channels_y, self.out_channels_y, 3, padding=8, dilation=8)
        self.conv5_y = nn.Conv1d(self.out_channels_y, self.out_channels_y, 3, padding=16, dilation=16)

        #self.bottle1_x = nn.Conv1d(self.out_channels, self.out_channels, 1)
        #self.bottle2_x = nn.Conv1d(self.out_channels, self.out_channels, 1)
        #self.bottle3_x = nn.Conv1d(self.out_channels, self.out_channels, 1)
        #self.bottle4_x = nn.Conv1d(self.out_channels, self.out_channels, 1)
        #self.bottle5_x = nn.Conv1d(self.out_channels, self.out_channels, 1)
        self.bottle6_x = nn.Conv1d(self.out_channels, self.num_classes, 1)
        #self.bottle1_y = nn.Conv1d(self.out_channels_y, self.out_channels_y, 1)
        #self.bottle2_y = nn.Conv1d(self.out_channels_y, self.out_channels_y, 1)
        #self.bottle3_y = nn.Conv1d(self.out_channels_y, self.out_channels_y, 1)
        #self.bottle4_y = nn.Conv1d(self.out_channels_y, self.out_channels_y, 1)
        #self.bottle5_y = nn.Conv1d(self.out_channels_y, self.out_channels_y, 1)

        self.bottle11_y = nn.Conv1d(self.out_channels_y, self.out_channels, 1)
        self.bottle22_y = nn.Conv1d(self.out_channels_y, self.out_channels, 1)
        self.bottle33_y = nn.Conv1d(self.out_channels_y, self.out_channels, 1)
        self.bottle44_y = nn.Conv1d(self.out_channels_y, self.out_channels, 1)
        self.bottle55_y = nn.Conv1d(self.out_channels_y, self.out_channels, 1)
        self.bottle6_y = nn.Conv1d(self.out_channels_y, self.num_classes, 1)

    def forward(self, x, y, mask_r,mask_s):
        out0_x = self.bottle0_x(x)
        out0_y = self.bottle0_y(y)

        out11_y = F.relu(self.conv1_y(out0_y))
        # out1_y = self.bottle1_y(out1_y)
        out1_y = self.Drop1(out11_y)
        out1_y = (out0_y + out1_y) * mask_s[:, 0:1, :]

        out1_x = F.relu(self.conv1_x(out0_x))
        attention1=F.sigmoid(self.bottle11_y(out11_y))
        out1_x= attention1*out1_x #hadamard productprint
        # out1_x = self.bottle1_x(out1_x)
        out1_x = self.Drop1(out1_x)
        out1_x = (out0_x + out1_x) * mask_r[:, 0:1, :]


        out22_y = F.relu(self.conv2_y(out1_y))
        # out1_y = self.bottle1_y(out1_y)
        out2_y = self.Drop2(out22_y)
        out2_y = (out1_y + out2_y) * mask_s[:, 0:1, :]

        out2_x = F.relu(self.conv2_x(out1_x))
        attention2 = F.sigmoid(self.bottle22_y(out22_y))
        out2_x= attention2*out2_x #hadamard product
        # out1_x = self.bottle1_x(out1_x)
        out2_x = self.Drop2(out2_x)
        out2_x = (out1_x + out2_x) * mask_r[:, 0:1, :]


        out33_y = F.relu(self.conv3_y(out2_y))
        # out1_y = self.bottle1_y(out1_y)
        out3_y = self.Drop3(out33_y)
        out3_y = (out2_y + out3_y) * mask_s[:, 0:1, :]

        out3_x = F.relu(self.conv3_x(out2_x))
        attention3 = F.sigmoid(self.bottle33_y(out33_y))
        out3_x= attention3*out3_x #hadamard product
        # out1_x = self.bottle1_x(out1_x)
        out3_x = self.Drop3(out3_x)
        out3_x = (out2_x + out3_x) * mask_r[:, 0:1, :]



        out44_y = F.relu(self.conv4_y(out3_y))
        # out1_y = self.bottle1_y(out1_y)
        out4_y = self.Drop4(out44_y)
        out4_y = (out3_y + out4_y) * mask_s[:, 0:1, :]

        out4_x = F.relu(self.conv4_x(out3_x))
        attention4 = F.sigmoid(self.bottle44_y(out44_y))
        out4_x= attention4*out4_x #hadamard product
        # out1_x = self.bottle1_x(out1_x)
        out4_x = self.Drop4(out4_x)
        out4_x = (out3_x + out4_x) * mask_r[:, 0:1, :]

        # print(out4_x.size())
        out55_y = F.relu(self.conv5_y(out4_y))
        # out1_y = self.bottle1_y(out1_y)
        # out5_y = self.Drop5(out55_y)
        # out5_y = (out4_y + out5_y) * mask_s[:, 0:1, :]

        out5_x = F.relu(self.conv5_x(out4_x))
        attention5 = F.sigmoid(self.bottle55_y(out55_y))
        out5_x= attention5*out5_x #hadamard product
        # out1_x = self.bottle1_x(out1_x)
        out5_x = self.Drop5(out5_x)
        out5_x = (out4_x + out5_x) * mask_r[:, 0:1, :]

        out6_r= self.bottle6_x(out5_x)* mask_r[:, 0:1, :]
        # out6_s = self.bottle6_y(out5_y) * mask_s[:, 0:1, :]

        mask1=torch.einsum('bct -> bt', attention1)
        mask2 = torch.einsum('bct -> bt', attention2)
        mask3 = torch.einsum('bct -> bt', attention3)
        mask4 = torch.einsum('bct -> bt', attention4)
        mask5 = torch.einsum('bct -> bt', attention5)


        return out6_r
    #mask1,mask2,mask3,mask4,mask5 #out6_s
