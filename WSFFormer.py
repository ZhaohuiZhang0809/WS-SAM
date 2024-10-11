import torch
import torch.nn as nn
from einops import rearrange
from thop import profile
from torchinfo import summary
from models.common import Dwt2d, Iwt2d, Mlp, MultiHeadAttention, DWConv


# class WSFAttention(nn.Module):
#     def __init__(self, in_channels, patch_size, num_heads=4):
#         super(WSFAttention, self).__init__()
#         self.H = patch_size
#         self.W = patch_size
#         self.num_heads = num_heads
#         head_dim = in_channels // self.num_heads
#         self.scale = head_dim ** -0.5
#
#         self.dwt = Dwt2d()
#         self.iwt = Iwt2d()
#
#         self.q_proj = nn.Sequential(
#             nn.Linear(in_channels, in_channels // 4),
#             nn.LayerNorm(in_channels // 4),
#             nn.ReLU(inplace=True)
#         )
#         self.kv_proj = nn.Sequential(
#             nn.Linear(in_channels, in_channels // 2),
#             nn.LayerNorm(in_channels // 2),
#             nn.ReLU(inplace=True)
#         )
#
#         self.softmax = nn.Softmax(dim=-1)
#
#         self.out_proj = nn.Linear(in_channels, 4 * in_channels)
#
#         self.CA = CAFormer(dim=in_channels, num_heads=4, mlp_ratio=0.5)
#
#     def forward(self, sf, wf):
#         B, L, C = sf.shape
#         H, W = self.H, self.W
#
#         t_sf = self.q_proj(sf)
#         t_wf = self.kv_proj(wf)
#
#         # 转入频域
#         t_sf = t_sf.view(B, H, W, C // 4).permute(0, 3, 1, 2)
#         t_wf = t_wf.view(B, H, W, C // 2).permute(0, 3, 1, 2)
#         t_sf = self.dwt(t_sf).permute(0, 2, 3, 1).view(B, L // 4, C)
#         t_wf = self.dwt(t_wf).permute(0, 2, 3, 1).view(B, L // 4, C * 2)
#
#         query = t_sf
#         key, value = t_wf.chunk(2, dim=-1)
#
#         # 多头划分
#         query = rearrange(query, 'b n (n_h head_dim) -> b n_h n head_dim', n_h=self.num_heads)
#         key = rearrange(key, 'b n (n_h head_dim) -> b n_h n head_dim', n_h=self.num_heads)
#         value = rearrange(value, 'b n (n_h head_dim) -> b n_h head_dim n', n_h=self.num_heads)
#
#         att = torch.matmul(query, key.permute(0, 1, 3, 2)) * self.scale
#         att = self.softmax(att)
#
#         att = torch.matmul(value, att)
#         x = rearrange(att, 'b n_h head_dim (h w)-> b (h w) (n_h head_dim)', n_h=self.num_heads, h=H//2)
#         x = self.out_proj(x)
#         x = rearrange(x, 'b (h w) c -> b c h w', h=H//2)
#         x = self.iwt(x)
#         x = rearrange(x, 'b c h w-> b (h w) c')
#
#         out = self.CA(q_x=x, kv_x=sf)
#
#         return out


class WSFAttention(nn.Module):
    def __init__(self, in_channels, patch_size, num_heads=3):
        super(WSFAttention, self).__init__()
        self.H = patch_size
        self.W = patch_size
        self.num_heads = num_heads
        head_dim = in_channels // self.num_heads
        self.scale = head_dim ** -0.5

        self.dwt = Dwt2d()
        self.iwt = Iwt2d()

        self.q_proj = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.LayerNorm(in_channels)
        )
        self.kv_proj = nn.Sequential(
            nn.Linear(in_channels, in_channels * 2),
            nn.LayerNorm(in_channels * 2)
        )

        self.softmax = nn.Softmax(dim=-1)

        self.out_proj = nn.Linear(in_channels, in_channels)



    def forward(self, sf, wf):
        B, L, C = sf.shape
        H, W = self.H, self.W

        t_sf = self.q_proj(sf)
        t_wf = self.kv_proj(wf)

        # 转入频域
        t_sf = t_sf.view(B, H, W, C).permute(0, 3, 1, 2)
        t_wf = t_wf.view(B, H, W, C * 2).permute(0, 3, 1, 2)
        t_sf = self.dwt(t_sf).permute(0, 2, 3, 1).view(B, L // 4, C * 4)
        t_wf = self.dwt(t_wf).permute(0, 2, 3, 1).view(B, L // 4, C * 8)

        query = t_sf
        key, value = t_wf.chunk(2, dim=-1)

        q1 = query[:, :, 0: C]
        q2 = query[:, :, C: C * 4]

        k1 = key[:, :, 0: C]
        k2 = key[:, :, C: C * 4]

        v1 = value[:, :, 0: C]
        v2 = value[:, :, C: C * 4]

        # 高频运算(HH)
        att1 = torch.matmul(q1, k1.permute(0, 2, 1))
        att1 = self.softmax(att1)
        x1 = torch.matmul(att1, v1)
        x1 = rearrange(x1, 'b (h w) c -> b c h w', h=H//2)


        # 多头划分 -> 低频运算 (HL,LH,LL)
        q2 = rearrange(q2, 'b n (n_h head_dim) -> b n_h n head_dim', n_h=self.num_heads)
        k2 = rearrange(k2, 'b n (n_h head_dim) -> b n_h n head_dim', n_h=self.num_heads)
        v2 = rearrange(v2, 'b n (n_h head_dim) -> b n_h head_dim n', n_h=self.num_heads)

        att2 = torch.matmul(q2, k2.permute(0, 1, 3, 2)) * self.scale
        att2 = self.softmax(att2)
        x2 = torch.matmul(v2, att2)
        x2 = rearrange(x2, 'b n_h head_dim (h w)-> b (n_h head_dim) h w', n_h=self.num_heads, h=H//2)

        x = torch.cat([x1, x2], dim=1)
        x = self.iwt(x)
        x = rearrange(x, 'b c h w-> b (h w) c')

        out = self.out_proj(x)

        return out



class WSFFormer(nn.Module):
    def __init__(self, in_channels, patch_size):
        super(WSFFormer, self).__init__()
        self.norm = nn.LayerNorm(in_channels)
        self.wsf_attn = WSFAttention(in_channels, patch_size)
        self.patch_size = patch_size
        self.mlp = Mlp(in_channels)

        # self.CA = CAFormer(dim=in_channels, num_heads=4, mlp_ratio=0.5)
        self.CA = MultiHeadAttention(dim=in_channels, num_heads=4,)

        self.Voting_Unit = nn.Sequential(
            DWConv(in_channels, 1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

    def forward(self, sf, wf):
        H = self.patch_size
        attn = self.norm(self.wsf_attn(sf, wf)) + sf
        out = self.mlp(self.norm(attn)) + attn
        out = self.mlp(self.CA(q_x=out, kv_x=sf)) + out
        out = rearrange(out, 'b (h w) c -> b c h w', h=H)
        v = self.Voting_Unit(out)
        sf = rearrange(sf, 'b (h w) c -> b c h w', h=H)
        out = sf * v
        out = rearrange(out, 'b c h w -> b (h w) c', h=H)

        return self.norm(nn.ReLU()(out))


if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net = WSFFormer(in_channels=768, patch_size=10).to(device)

    inputs1 = torch.randn(1, 100, 768).to(device)
    inputs2 = torch.randn(1, 100, 768).to(device)
    flops, params = profile(net, (inputs1, inputs2,))
    print("FLOPs=, params=", flops, params)
    print("FLOPs=", str(flops / 1e9) + '{}'.format("G"))
    print("params=", str(params / 1e6) + '{}'.format("M"))


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = WSFFormer(in_channels=768, patch_size=10).to(device)
    summary(net, [(1, 100, 768), (1, 100, 768)])
