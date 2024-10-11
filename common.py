import torch
import torch.nn as nn
import pywt
from einops import rearrange
from timm.layers import to_2tuple
import math

from torch import Tensor


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class CVFM(nn.Module):
    r"""Channel voting fusion module"""
    def __init__(self, in_channels:int, ratio=8):
        super(CVFM, self).__init__()

        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1, bias=False)
        self.softmax = nn.Softmax(dim=1)

        self.channel_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False)

        self.channel_trans_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//ratio, kernel_size=1),
            nn.LayerNorm([in_channels//ratio, 1, 1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels//ratio, in_channels, kernel_size=1)
        )

        self.voting_gate = nn.Sequential(
            DWConv(in_channels,1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        B, C, H, W = x1.shape
        L = H * W

        content_feature1 = self.conv(x1).view(B, -1, L).permute(0, 2, 1)  # HW x 1
        content_feature1 = self.softmax(content_feature1)
        channel_feature1 = self.channel_conv(x1).view(B, -1, L)  # C x HW
        channel_pooling1 = torch.bmm(channel_feature1, content_feature1).view(B, -1, 1, 1)  # C x 1 x 1
        channel_weight1 = self.channel_trans_conv(channel_pooling1)

        content_feature2 = self.conv(x2).view(B, -1, L).permute(0, 2, 1)  # HW x 1
        content_feature2 = self.softmax(content_feature2)
        channel_feature2 = self.channel_conv(x2).view(B, -1, L)  # C x HW
        channel_pooling2 = torch.bmm(channel_feature2, content_feature2).view(B, -1, 1, 1)  # C x 1 x 1
        channel_weight2 = self.channel_trans_conv(channel_pooling2)

        channel_weight = channel_weight1 + channel_weight2

        x1 = x1 * channel_weight

        v = self.voting_gate(x1)
        return v


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)


class DoubleLinear(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleLinear, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(in_ch, out_ch),
            nn.LayerNorm(out_ch),
            nn.ReLU(inplace=True),
            nn.Linear(out_ch, out_ch),
            nn.LayerNorm(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.linear(input)


class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding
    Args:
        img_size (int): Image size.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        # reshape
        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # shape = (B, H/2, W/2, C)
        x1 = x[:, 1::2, 0::2, :]  # shape = (B, H/2, W/2, C)
        x2 = x[:, 0::2, 1::2, :]  # shape = (B, H/2, W/2, C)
        x3 = x[:, 1::2, 1::2, :]  # shape = (B, H/2, W/2, C)

        x = torch.cat([x0, x1, x2, x3], -1)  # shape = (B, H/2, W/2, 4*C)
        x = x.view(B, -1, 4 * C)  # shape = (B, H*W/4, 4*C)

        x = self.norm(x)
        x = self.reduction(x)  # shape = (B, H*W/4, 2*C)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops


# 2D DWT
class Dwt2d(nn.Module):
    def __init__(self):
        super(Dwt2d, self).__init__()
        self.requires_grad = False

    def dwt(self, x):
        with torch.no_grad():
            x = x.cpu()    ##
            # LL, (LH, HL, HH) = pywt.dwt2(x, 'haar')
            LL, (LH, HL, HH) = pywt.dwt2(x, 'haar')

            LL = torch.tensor(LL).cuda()
            LH = torch.tensor(LH).cuda()
            HL = torch.tensor(HL).cuda()
            HH = torch.tensor(HH).cuda()

        return torch.cat((LL, LH, HL, HH), 1)

    def forward(self, x):
        out = self.dwt(x)
        return out


# 2D IWT
class Iwt2d(nn.Module):
    def __init__(self):
        super(Iwt2d, self).__init__()
        self.requires_grad = False

    def iwt(self, x):
        with torch.no_grad():
            in_batch, in_channel, in_height, in_width = x.size()
            ch = in_channel // 4
            LL = x[:, 0: ch, :, :]
            LH = x[:, ch: ch * 2, :, :]
            HL = x[:, ch * 2: ch * 3, :, :]
            HH = x[:, ch * 3: ch * 4, :, :]

            coeffs = LL.cpu(), (LH.cpu(), HL.cpu(), HH.cpu())

            x = pywt.idwt2(coeffs, 'haar')
            # x = pywt.idwt2(coeffs, 'db2')
            x = torch.tensor(x).cuda()

        return x

    def forward(self, x):
        out = self.iwt(x)
        return out


class DWConv(nn.Module):
    r"""DepthWise Separable Convolution"""
    def __init__(self, in_channels, out_channels, stride=1, padding=2, dilation=2, expand_ratio=4):
        super(DWConv, self).__init__()

        self.DWConv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, stride, padding, dilation, groups=in_channels),         # Depth-wise Convolution
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, 1),                                                       # Point-wise Convolution
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.DWConv(x)

        return x


class Dwt(nn.Module):
    def __init__(self):
        super(Dwt, self).__init__()
        self.requires_grad = False

    def dwt(self, x):
        with torch.no_grad():
            x = x.cpu()    ##
            # LL, (LH, HL, HH) = pywt.dwt2(x, 'haar')
            LL, (LH, HL, HH) = pywt.dwt2(x, 'haar')
            LL = torch.tensor(LL).cuda()
            LH = torch.tensor(LH).cuda()
            HL = torch.tensor(HL).cuda()
            HH = torch.tensor(HH).cuda()

        return LL, LH, HL, HH

    def forward(self, x):
        out = self.dwt(x)
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout)

    def forward(self, q_x, kv_x, attn_mask=None):
        out, weight = self.multihead_attn(query=q_x, key=kv_x, value=kv_x, attn_mask=attn_mask)

        return out


class MultiCrossAttention(nn.Module):
    """
    An attention layer that allows for downscaling the size of the embedding
    after projection to queries, keys, and values.
    """

    def __init__(self, embedding_dim: int, num_heads: int, downsample_rate: int = 1 ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert self.internal_dim % num_heads == 0, "num_heads must divide embedding_dim."

        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

    def _recombine_heads(self, x: Tensor) -> Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C

    def forward(self, q: Tensor, kv: Tensor) -> Tensor:
        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(kv)
        v = self.v_proj(kv)

        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        # Attention
        _, _, _, c_per_head = q.shape
        attn = q @ k.permute(0, 1, 3, 2)  # B x N_heads x N_tokens x N_tokens
        attn = attn / math.sqrt(c_per_head)
        attn = torch.softmax(attn, dim=-1)

        # Get output
        out = attn @ v
        out = self._recombine_heads(out)
        out = self.out_proj(out)

        return out


class CAFormer(nn.Module):
    r""" Token2ImageFormer/Image2TokenFormer """

    def __init__(self, dim, mlp_ratio=1, drop=0., act_layer=nn.GELU, num_heads=3):
        super(CAFormer, self).__init__()

        self.CAttn = MultiCrossAttention(dim, num_heads=num_heads)
        self.norm = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, q_x, kv_x):
        x = self.norm(self.CAttn(q_x, kv_x)) + q_x
        x = self.norm(self.mlp(x)) + x

        return x


class PatchExpand(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.expand = nn.Linear(dim, 2*dim, bias=False) if dim_scale==2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=C//4)
        x = x.view(B,-1,C//4)
        x= self.norm(x)

        return x


class FinalPatchExpand_X4(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, 16 * dim, bias=False)
        self.output_dim = dim
        self.norm = norm_layer(self.output_dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale, c=C//(self.dim_scale**2))
        x = x.view(B,-1,self.output_dim)
        x= self.norm(x)

        return x


# if __name__ == '__main__':
#     random_data1 = torch.rand(2, 200, 4)
#     random_data2 = torch.rand(2, 100, 4)
#     CA = CrossAttention(4, 4)
#     out = CA(random_data1, random_data2)
#     print(out)
