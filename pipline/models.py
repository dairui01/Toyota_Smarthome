import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.init as init




class SDTCN(nn.Module):
    def __init__(self, inter_dim=256, input_dim=1024, num_classes=157):
        super(Dilated_TCN, self).__init__()
        self.inter_dim = int(inter_dim)
        self.input_dim = int(input_dim)
        self.num_classes = int(num_classes)

        self.conv1 = nn.Conv1d(self.inter_dim, self.inter_dim, 3, padding=1, dilation=1)
        self.conv2 = nn.Conv1d(self.inter_dim, self.inter_dim, 3, padding=2, dilation=2)
        self.conv3 = nn.Conv1d(self.inter_dim, self.inter_dim, 3, padding=4, dilation=4)
        self.conv4 = nn.Conv1d(self.inter_dim, self.inter_dim, 3, padding=8, dilation=8)
        self.conv5 = nn.Conv1d(self.inter_dim, self.inter_dim, 3, padding=16, dilation=16)

        self.Drop1 = nn.Dropout()
        self.Drop2 = nn.Dropout()
        self.Drop3 = nn.Dropout()
        self.Drop4 = nn.Dropout()
        self.Drop5 = nn.Dropout()

        self.bottle0 = nn.Conv1d(self.input_dim, self.inter_dim, 1)
        self.bottle1 = nn.Conv1d(self.inter_dim, self.inter_dim, 1)
        self.bottle2 = nn.Conv1d(self.inter_dim, self.inter_dim, 1)
        self.bottle3 = nn.Conv1d(self.inter_dim, self.inter_dim, 1)
        self.bottle4 = nn.Conv1d(self.inter_dim, self.inter_dim, 1)
        self.bottle5 = nn.Conv1d(self.inter_dim, self.inter_dim, 1)
        self.bottle6 = nn.Conv1d(self.inter_dim, self.num_classes, 1)

    def forward(self, x, mask):
        out0 = self.bottle0(x)

        out1 = F.relu(self.conv1(out0))
        out1 = self.bottle1(out1)
        
        out1 = self.Drop1(out1)
        out1 = (out0 + out1) * mask[:, 0:1, :]

        out2 = F.relu(self.conv2(out1))
        out2 = self.bottle2(out2)
        out2 = self.Drop2(out2)
        out2 = (out1 + out2) * mask[:, 0:1, :]
      
        out3 = F.relu(self.conv3(out2))
        out3 = self.bottle3(out3)
        out3 = self.Drop3(out3)
        out3 = (out2 + out3) * mask[:, 0:1, :]
        
        out4 = F.relu(self.conv4(out3))
        out4 = self.bottle4(out4)
        out4 = self.Drop4(out4)
        out4 = (out3 + out4) * mask[:, 0:1, :]
        
        out5 = F.relu(self.conv5(out4))
        out5 = self.bottle5(out5)
        out5 = self.Drop5(out5)
        out5 = (out4 + out5) * mask[:, 0:1, :]
        
        out6 = self.bottle6(out5) * mask[:, 0:1, :] 

        return out6



