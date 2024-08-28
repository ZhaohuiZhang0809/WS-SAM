import torch.nn as nn
import torch
from thop import profile
from torchsummary import summary

from models.common import Dwt2d, Iwt2d, DWConv, DoubleConv


# class fWavelet(nn.Module):
#     def __init__(self, in_channels, out_channels, downsample=True):
#         super().__init__()
#
#         self.downsample = downsample
#
#         if downsample:
#             self.downsample = nn.AvgPool2d(2)
#
#         self.dwt = Dwt2d()
#
#         self.DWConv1 = DWConv(in_channels * 4, out_channels)
#         self.DWConv2 = DWConv(in_channels, out_channels)
#
#     def forward(self, input):
#         out_down = None
#
#         if self.downsample:
#             out_down = self.downsample(input)
#
#         out_dwt = self.dwt(input)
#         out = self.DWConv1(out_dwt) + self.DWConv2(out_down)
#
#         return out
#
#
# class Wavelet_Encoder(nn.Module):
#     def __init__(self, in_channels, embed_dim=96, downsample=True):
#         super().__init__()
#
#         self.dconv1 = nn.Sequential(
#             DWConv(in_channels, embed_dim),
#             nn.MaxPool2d(2)
#         )
#
#         self.dconv2 = nn.Sequential(
#             DWConv(embed_dim, embed_dim // 2),
#             nn.MaxPool2d(2)
#         )
#
#         self.wconv1 = nn.Sequential(
#             DWConv(embed_dim // 2, embed_dim // 2),
#             Dwt2d(),
#         )
#
#         self.wconv2 = nn.Sequential(
#             DWConv(embed_dim * 2, embed_dim),
#             Dwt2d(),
#         )
#
#     def forward(self, x):
#
#         x1 = self.dconv1(x)
#         x2 = self.dconv2(x1)
#         x3 = self.wconv1(x2)
#         x4 = self.wconv2(x3)
#
#         return x4


class Wavelet_Encoder(nn.Module):
    def __init__(self, in_channels, embed_dim=96, downsample=True):
        super().__init__()

        self.dwt = Dwt2d()

        self.conv1 = DWConv(in_channels, embed_dim // 4)
        self.conv2 = DWConv(embed_dim, embed_dim // 2)
        self.conv3 = DWConv(embed_dim * 2, embed_dim)
        self.conv4 = DWConv(embed_dim * 4, embed_dim * 2)
        self.conv5 = DWConv(embed_dim * 8, embed_dim * 4)

    def forward(self, x):
        wx_list = []
        c1 = self.conv1(x)
        p1 = self.dwt(c1)
        c2 = self.conv2(p1)
        p2 = self.dwt(c2)
        c3 = self.conv3(p2)
        w3 = self.dwt(c3)
        c4 = self.conv4(w3)
        w4 = self.dwt(c4)
        c5 = self.conv5(w4)

        wx_list.append(c1)  # 24
        wx_list.append(c2)  # 48
        wx_list.append(c3)  # 96
        wx_list.append(c4)  # 192

        return c5, wx_list

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = Wavelet_Encoder(in_channels=1).to(device)
    # 打印网络结构和参数
    summary(net, (1, 320, 320))

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = Wavelet_Encoder(in_channels=1).to(device)
    inputs = torch.randn(1, 1, 320, 320).to(device)
    flops, params = profile(net, (inputs,))
    print("FLOPs=, params=", flops, params)
    print("FLOPs=", str(flops / 1e9) + '{}'.format("G"))
    print("params=", str(params / 1e6) + '{}'.format("M"))
