import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from thop import profile
from timm.layers import DropPath, to_2tuple
from torch.utils import checkpoint
from torchsummary import summary

from models.common import Mlp, PatchEmbed, PatchMerging, DWConv


class PCWAttention(nn.Module):
    r""" Pixel-Channel Windows hybrid Attention """

    def __init__(self, dim, input_resolution, num_heads=8, window_size=3, qk_scale=None, qkv_bias=True, proj_drop=0.):
        super().__init__()

        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = qk_scale or self.head_dim ** -0.5
        self.local_len = window_size ** 2

        self.relative_pos_bias_local = nn.Parameter(
            nn.init.trunc_normal_(torch.empty(num_heads, self.local_len), mean=0,
                                  std=0.0004))

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        # Partition window
        self.unfold = nn.Unfold(kernel_size=window_size, padding=window_size // 2, stride=1)

        self.pool_c = nn.AdaptiveAvgPool2d((1, 1))
        self.softmax = nn.Softmax(dim=-2)

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        q_pixel = F.normalize(self.q(x).reshape(B, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3),
                              dim=-1) * self.scale
        k_local, v_local = self.kv(x).chunk(2, dim=-1)
        k_local = F.normalize(k_local.reshape(B, L, self.num_heads, self.head_dim), dim=-1).reshape(B, L, -1)
        kv_local = torch.cat([k_local, v_local], dim=-1).permute(0, 2, 1).reshape(B, -1, H, W)
       k_local, v_local = (self.unfold(kv_local).reshape(B, 2 * self.num_heads, self.head_dim, self.local_len, L)
                            .permute(0, 1, 4, 2, 3).chunk(2, dim=1))    # b, h_n, global, h_d, local

        attn_local = (q_pixel.unsqueeze(-2) @ k_local).squeeze(-2)      # b, h_n, global, local

        _x = rearrange(x, 'B (H W) C ->  B C H W', H=H)
        q_pool = rearrange(self.pool_c(_x), 'B (n_h h_dim) H W -> B n_h (H W) h_dim', n_h=self.num_heads)
        q_pool = q_pool.expand(-1, -1, L, -1).unsqueeze(-2)             # b, h_n, global, 1, h_d

        attn_pool = (q_pool @ k_local).squeeze(-2)                      # b, h_n, global, local

        attn_score = attn_local + attn_pool + self.relative_pos_bias_local.unsqueeze(1)     # b, h_n, global, local
        attn_score = self.softmax(attn_score).unsqueeze(-2).permute(0, 1, 2, 4, 3)          # b, h_n, global, local, 1

        x = rearrange((v_local @ attn_score).squeeze(-1), 'B n_h L h_d -> B L (n_h h_d)', n_h=self.num_heads)   # b, h_n, global, h_d, 1 -> b, global, c

        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class PCWFormerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (int): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        fused_window_process (bool, optional): If True, use one kernel to fused window shift & window partition for acceleration, similar for the reversed part. Default: False
    """

    def __init__(self, dim, input_resolution, num_heads=3, window_size=3,
                 mlp_ratio=4, qkv_bias=True, qk_scale=None, drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()

        # Partition window
        self.unfold = nn.Unfold(kernel_size=window_size, padding=window_size // 2, stride=1)

        # strip pooling
        self.pool_h = nn.AdaptiveAvgPool2d((1, None))
        self.pool_w = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_c = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(dim, dim // mlp_ratio, bias=False),
            nn.ReLU(),
            nn.Linear(dim // mlp_ratio, dim, bias=False),
            nn.Sigmoid()
        )

        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)

        self.norm1 = norm_layer(dim)

        self.attn = PCWAttention(dim, input_resolution=input_resolution, window_size=self.window_size,
                                 num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        # self.spatial_interaction = nn.Sequential(
        #     nn.Conv2d(dim, dim // 16, kernel_size=1),
        #     nn.BatchNorm2d(dim // 16),
        #     nn.GELU(),
        #     nn.Conv2d(dim // 16, 1, kernel_size=1)
        # )

        self.DWConv = DWConv(1, 1)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        shortcut = x

        _x = self.norm1(x)

        # SPA
        sp_x = rearrange(_x, 'B (H W) C ->  B C H W ', H=H)
        pool_h = self.pool_h(sp_x).view(B, C, H)
        pool_w = self.pool_w(sp_x).view(B, C, W)
        pool_c = self.fc(self.pool_c(sp_x).view(B, C)).view(B, 1, C)
        spatial_affinity = torch.bmm(pool_h.permute(0, 2, 1), pool_w).view(B, 1, H, W)
        spatial_weight_M = rearrange(nn.Sigmoid()(self.DWConv(spatial_affinity)), 'B C H W  -> B (H W) C', H=H)
        # spatial_attened_x = spatial_attn.expand_as(sp_x) * pool_c * sp_x + sp_x

        # PCWAttention
        pcw_attn_x = self.attn(x)

        x = pcw_attn_x * spatial_weight_M * pool_c

        x = shortcut + self.drop_path(x)

        # FFN
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (int): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4, qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            PCWFormerBlock(dim=dim, input_resolution=input_resolution,
                           num_heads=num_heads, window_size=window_size,
                           mlp_ratio=mlp_ratio,
                           qkv_bias=qkv_bias, qk_scale=qk_scale,
                           drop=drop,
                           drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                           norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        x_ = None
        for blk in self.blocks:
            if self.use_checkpoint:
                x_ = checkpoint.checkpoint(blk, x)
            else:
                x_ = blk(x)

        if self.downsample is not None:
            x = self.downsample(x_)
        else:
            x = x_

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops


class Image_Encoder(nn.Module):
    r"""
    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (int): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
        fused_window_process (bool, optional): If True, use one kernel to fused window shift & window partition for acceleration, similar for the reversed part. Default: False
    """

    def __init__(self, img_size=320, patch_size=4, in_chans=3, num_classes=1,
                 embed_dim=96, depths=[6, 2, 2], num_heads=[3, 6, 12],
                 window_size=3, mlp_ratio=4, qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, fused_window_process=False, **kwargs):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            nn.init.trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               # downsample=PatchMerging,
                               use_checkpoint=use_checkpoint)
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        i_layer = 0
        for layer in self.layers:
            x = layer(x)
            # x_ = rearrange(x_, 'b (h w) c -> b c h w', h=self.patches_resolution[0] // (2 ** i_layer))
            # fx.append(x_)
            i_layer += 1

        return x

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
        flops += self.num_features * self.num_classes
        return flops


# if __name__ == '__main__':
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     net = Image_Encoder(in_chans=1).to(device)
#     summary(net, (1, 320, 320), batch_size=2)

if __name__ == "__main__":
    net = Image_Encoder(in_chans=1)
    inputs = torch.randn(1, 1, 320, 320)
    flops, params = profile(net, (inputs,))
    print("FLOPs=, params=", flops, params)
    print("FLOPs=", str(flops / 1e9) + '{}'.format("G"))
    print("params=", str(params / 1e6) + '{}'.format("M"))
