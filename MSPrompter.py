import numpy as np
import torch
from einops import rearrange
from sklearn.cluster import KMeans
from thop import profile
from torch import nn
import torch.nn.functional as F
from torchinfo import summary

from models.common import PatchEmbed, DoubleConv, MultiCrossAttention, Mlp
from models.Image_Encoder import PCWFormerBlock, Image_Encoder


class MSPrompter(nn.Module):
    def __init__(self, dim=96, ref_chans=1, img_size=320, patch_size=4, embed_dim=96, norm_layer=nn.LayerNorm,
                 num_heads=3,
                 window_size=3, mlp_ratio=4, qkv_bias=True, qk_scale=None, drop=0., drop_path=0.,
                 simple_encoder=False):
        super(MSPrompter, self).__init__()

        self.simple_encoder = simple_encoder

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=ref_chans, embed_dim=embed_dim,
            norm_layer=norm_layer)

        patches_resolution = self.patch_embed.patches_resolution

        self.pcwformer = PCWFormerBlock(dim=96, input_resolution=(patches_resolution[0],
                                                                  patches_resolution[1]),
                                        num_heads=num_heads, window_size=window_size,
                                        mlp_ratio=mlp_ratio,
                                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                                        drop=drop,
                                        drop_path=drop_path,
                                        norm_layer=norm_layer)

        self.encoder = nn.Sequential(
            DoubleConv(1, embed_dim // 2),
            nn.MaxPool2d(2),
            DoubleConv(embed_dim // 2, embed_dim),
            nn.MaxPool2d(2),
            DoubleConv(embed_dim, embed_dim)
        )

        self.image_encoder = Image_Encoder(img_size=img_size, patch_size=4, in_chans=1, num_classes=1, window_size=3,
                                           depths=[4], num_heads=[3])

        self.DMPromptGen = DMPromptGen(dim, img_size)

        self.norm = nn.LayerNorm(dim)

    def forward(self, query_x, reference_x, reference_p):
        if self.simple_encoder:
            r_x = self.encoder(reference_x)
            r_x = rearrange(r_x, 'b c h w ->  b (h w) c')
        else:
            r_x = self.image_encoder(reference_x)

        e_q2p, e_p3 = self.DMPromptGen(query_x, r_x, reference_p)

        return e_q2p, e_p3


class MaskAvgPool2d(nn.Module):
    def __init__(self,):
        super(MaskAvgPool2d, self).__init__()

    def forward(self, x, mask):
        masked_x = x * mask

        prototype = F.adaptive_avg_pool2d(masked_x, (1, 1))
        return prototype


class DMPromptGen(nn.Module):
    """ Dual-path Meta-prompt Generation"""

    def __init__(self, dim, image_size, mlp_ratio=4, scaler=20):
        super(DMPromptGen, self).__init__()

        self.CAFormer = CAFormer(dim)
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        self.MAP = MaskAvgPool2d()
        self.MLP = Mlp(dim)
        self.scaler = scaler

        self.fc = nn.Sequential(
            nn.Linear(2 * dim, dim // mlp_ratio, bias=False),
            nn.ReLU(),
            nn.Linear(dim // mlp_ratio, dim, bias=False),
            nn.Sigmoid()
        )

    def Sim(self, fts, prototype, thresh):
        """
        Calculate the distance between features and prototypes

        Args:
            fts: input features
                expect shape: N x C x H x W
            prototype: prototype of one semantic class
                expect shape: 1 x C
        """
        sim = -F.cosine_similarity(fts, prototype[..., None, None], dim=1) * self.scaler
        coarse_mask = 1.0 - torch.sigmoid(0.5 * (sim - thresh))

        return coarse_mask    

    def forward(self, q_x, r_x, r_p):
        e_p1 = self.CAFormer(r_p, r_x)
        e_r2p = self.CAFormer(r_x, e_p1)
        e_q2r = self.CAFormer(q_x, r_x)

        p_r = self.avgpool(e_r2p.permute(0, 2, 1))

        e_q2r_ = rearrange(e_q2r, 'b (h w) c -> b c h w', h=self.size//4, w=self.size//4)
        thresh = self.MLP(e_q2r)
        thresh = rearrange(thresh, 'b (h w) c -> b c h w', h=self.size//4, w=self.size//4)
        c_mask = self.Sim(e_q2r_.unsqueeze(1), p_r, thresh)
        p_q = rearrange(self.MAP(e_q2r_, c_mask), 'b c h w -> b c (h w)')

        p_e = self.fc(
            torch.cat([
                p_r,
                p_q
                ], dim=1
            ).permute(0, 2, 1))

        # enhanced prototype
        e_q2r = e_q2r * p_e
        e_q2p = self.CAFormer(q_x, e_q2r)
        # e_q2p = q_x * p_e

        # update prompt
        e_p2 = self.CAFormer(e_p1, e_q2r)
        e_p3 = self.CAFormer(e_p2, q_x)

        return e_q2p, e_p3


class CAFormer(nn.Module):
    r""" Token2ImageFormer/Image2TokenFormer """

    def __init__(self, dim, mlp_ratio=4, drop=0., act_layer=nn.GELU):
        super(CAFormer, self).__init__()

        self.CAttn = MultiCrossAttention(dim, num_heads=3)
        self.norm = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, q_x, kv_x):
        x = self.norm(self.CAttn(q_x, kv_x)) + q_x
        x = self.norm(self.mlp(x)) + x

        return x


# if __name__ == '__main__':
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     net = MSPrompter(ref_chans=1, img_size=320).to(device)
#     summary(net, [(1, 6400, 96), (1, 1, 320, 320), (1, 11, 96)])

if __name__ == "__main__":
    net = MSPrompter(ref_chans=1, img_size=320)
    inputs_1 = torch.randn(1, 6400, 96)
    inputs_2 = torch.randn(1, 1, 320, 320)
    inputs_3 = torch.randn(1, 11, 96)
    flops, params = profile(net, (inputs_1, inputs_2, inputs_3))
    print("FLOPs=, params=", flops, params)
    print("FLOPs=", str(flops / 1e9) + '{}'.format("G"))
    print("params=", str(params / 1e6) + '{}'.format("M"))
