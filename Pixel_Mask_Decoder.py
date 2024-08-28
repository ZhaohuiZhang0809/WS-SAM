import numpy as np
import torch
from einops import rearrange
from sklearn.cluster import KMeans
from thop import profile
from torch import nn
import torch.nn.functional as F
from torchinfo import summary

from models.common import PatchEmbed, DoubleConv, MultiCrossAttention, Mlp, LayerNorm2d, MultiHeadAttention, DWConv


class CAFormer(nn.Module):
    r""" Token2ImageFormer/Image2TokenFormer """

    def __init__(self, dim, mlp_ratio=4, drop=0., act_layer=nn.GELU):
        super(CAFormer, self).__init__()

        self.CAttn = MultiCrossAttention(dim, num_heads=3)
        self.norm = nn.LayerNorm(dim)
        mlp_hidden_dim = dim * mlp_ratio
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, q_x, kv_x):
        x = self.norm(self.CAttn(q_x, kv_x)) + q_x
        x = self.norm(self.mlp(x)) + x

        return x


class MPIM(nn.Module):
    r""" Merging Prompt and Image Module """

    def __init__(self, dim, mlp_ratio=4, drop=0., act_layer=nn.GELU, out_classes=1):
        super(MPIM, self).__init__()

        self.CAttn = MultiCrossAttention(dim, num_heads=3)
        self.SA = MultiHeadAttention(dim, num_heads=3)
        self.norm = nn.LayerNorm(dim)

        mlp_hidden_dim = dim * mlp_ratio
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        feat_channels = dim
        self.query_feat = nn.Embedding(out_classes, feat_channels)

    def forward(self, image_x, prompt_x):
        B, L, C = image_x.shape
        x = self.norm(self.CAttn(image_x, prompt_x)) + image_x
        x = self.norm(self.mlp(x)) + x

        query_feat = self.query_feat.weight.unsqueeze(0).repeat((B, 1, 1))

        q_x = self.norm(self.CAttn(query_feat, x)) + query_feat
        q_x = self.norm(self.SA(q_x, q_x)) + q_x
        q_x = self.norm(self.mlp(q_x)) + q_x

        out = q_x @ image_x.permute(0, 2, 1)

        return out


class Pixel_Mask_Decoder(nn.Module):
    def __init__(self, dim, out_size, out_classes):
        super(Pixel_Mask_Decoder, self).__init__()
        self.out_size = out_size

        self.CAFormer = CAFormer(dim)
        self.MPIM = MPIM(dim // 4, out_classes, out_classes)

        # self.pixel_up = nn.Sequential(
        #     DoubleConv(dim, dim),
        #     nn.ConvTranspose2d(dim, dim, kernel_size=2, stride=2),
        #     LayerNorm2d(dim),
        #     DoubleConv(dim, dim // 2),
        #     nn.ConvTranspose2d(dim // 2, dim // 2, kernel_size=2, stride=2),
        #     LayerNorm2d(dim // 2)
        # )

        self.up3 = nn.ConvTranspose2d(dim, dim // 2, 2, stride=2)
        self.conv3 = DoubleConv(dim, dim // 2)
        self.up4 = nn.ConvTranspose2d(dim // 2, dim // 4, 2, stride=2)
        self.conv4 = DoubleConv(dim // 2, dim // 4)

        self.linear = nn.Linear(dim, dim // 4)
        self.norm = nn.BatchNorm2d(dim // 4)
        self.out_proj = nn.Conv2d(dim // 4, 1, 1)


    def forward(self, x, p, wx_list):
        H = self.out_size
        # upm
        up = self.CAFormer(p, x)
        up = self.linear(up)

        # MPIM
        x = rearrange(x, 'B (H W) C -> B C H W', H=H//4)
        # x_ = self.pixel_up(x)

        x3 = self.up3(x)
        x3 = torch.cat([wx_list[1], x3], dim=1)
        x3 = self.conv3(x3)

        x4 = self.up4(x3)
        x4 = torch.cat([wx_list[0], x4], dim=1)
        x_ = self.conv4(x4)

        x = rearrange(x_, 'B C H W -> B (H W) C', H=H)
        out = self.MPIM(x, up)
        out = rearrange(out, 'B C (H W) -> B C H W', H=H)

        out = self.out_proj(self.norm(out * x_))

        return out


if __name__ == "__main__":
    net = Pixel_Mask_Decoder(dim=96, out_size=320, out_classes=1)
    inputs1 = torch.randn(1, 6400, 96)
    inputs2 = torch.randn(1, 11, 96)
    inputs3 = torch.randn(1, 24, 320, 320)
    inputs4 = torch.randn(1, 48, 160, 160)
    flops, params = profile(net, (inputs1, inputs2,[inputs3, inputs4]))
    print("FLOPs=, params=", flops, params)
    print("FLOPs=", str(flops / 1e9) + '{}'.format("G"))
    print("params=", str(params / 1e6) + '{}'.format("M"))
