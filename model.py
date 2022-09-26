import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)

class Unet1D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Unet1D, self).__init__()
        self.ch = 128

        self.conv1 = DoubleConv(in_ch, self.ch)
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = DoubleConv(self.ch, self.ch*2)
        self.pool2 = nn.MaxPool1d(2)
        self.conv3 = DoubleConv(self.ch*2, self.ch*4)
        self.pool3 = nn.MaxPool1d(2)
        self.conv4 = DoubleConv(self.ch*4, self.ch*8)
        self.pool4 = nn.MaxPool1d(2)
        self.conv5 = DoubleConv(self.ch*8, self.ch*16)
        self.up6 = nn.ConvTranspose1d(self.ch*16, self.ch*8, 2, stride=2)
        self.conv6 = DoubleConv(self.ch*16, self.ch*8)
        self.up7 = nn.ConvTranspose1d(self.ch*8, self.ch*4, 2, stride=2)
        self.conv7 = DoubleConv(self.ch*8, self.ch*4)
        self.up8 = nn.ConvTranspose1d(self.ch*4, self.ch*2, 2, stride=2)
        self.conv8 = DoubleConv(self.ch*4, self.ch*2)
        self.up9 = nn.ConvTranspose1d(self.ch*2, self.ch, 2, stride=2)
        self.conv9 = DoubleConv(self.ch*2, self.ch)
        self.conv10 = nn.Conv1d(self.ch, out_ch, 1)

    def forward(self, x):
        #print(x.shape)
        c1 = self.conv1(x)
        p1 = self.pool1(c1)
        #print(p1.shape)
        c2 = self.conv2(p1)
        p2 = self.pool2(c2)
        #print(p2.shape)
        c3 = self.conv3(p2)
        p3 = self.pool3(c3)
        #print(p3.shape)
        c4 = self.conv4(p3)
        p4 = self.pool4(c4)
        #print(p4.shape)
        c5 = self.conv5(p4)
        up_6 = self.up6(c5)
        merge6 = torch.cat([up_6, c4], dim=1)
        c6 = self.conv6(merge6)
        up_7 = self.up7(c6)
        merge7 = torch.cat([up_7, c3], dim=1)
        c7 = self.conv7(merge7)
        up_8 = self.up8(c7)
        merge8 = torch.cat([up_8, c2], dim=1)
        c8 = self.conv8(merge8)
        up_9 = self.up9(c8)
        merge9 = torch.cat([up_9, c1], dim=1)
        c9 = self.conv9(merge9)
        c10 = self.conv10(c9)
        out = nn.Hardtanh()(c10)
        # out = c10
        return out


class GruRNN(nn.Module): 
    def __init__(self, input_size, hidden_size=1, num_layers=1):
        super().__init__()
 
        self.gru = nn.GRU(input_size, hidden_size, num_layers)
 
    def forward(self, _x):
        _x = _x.permute(2,0,1)
        x, _ = self.gru(_x)
        out = nn.Hardtanh()(x)
        return out.permute(1,2,0)


if __name__ == '__main__':
    m = GruRNN(88,88,10).cuda()
    # m = Unet1D(88,88).cpu()

    inp = torch.Tensor(np.zeros((1,88,20000))).cuda()
    outp = m(inp)
    pass
    # print(m)