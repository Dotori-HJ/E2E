from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce

from .pooler import DropPath, Mlp


def get_rel_pos(rel_pos, d):
    if isinstance(d, int):
        ori_d = rel_pos.shape[0]
        if ori_d == d:
            return rel_pos
        else:
            # Interpolate rel pos.
            new_pos_embed = F.interpolate(
                rel_pos.reshape(1, ori_d, -1).permute(0, 2, 1),
                size=d,
                mode="linear",
            )

            return new_pos_embed.reshape(-1, d).permute(1, 0)

def cal_rel_pos_temporal(attn, q, rel_pos_t):
    """
    Temporal Relative Positional Embeddings.
    """
    B, n_head, t, dim = q.size()

    dt = int(2 * t - 1)
    # Intepolate rel pos if needed.
    rel_pos_t = get_rel_pos(rel_pos_t, dt)

    # Scale up rel pos if shapes for q and k are different.
    dist_t = torch.arange(t)
    dist_t = dist_t[:, None] - dist_t[None, :]
    dist_t += (t - 1)
    Rt = rel_pos_t[dist_t.long()]

    r_q = q
    # [B, H, q_t, dim] -> [q_t, B, H, dim] -> [q_t, B*H, dim]
    r_q = r_q.permute(2, 0, 1, 3).reshape(t, B * n_head, dim)

    # [q_t, B*H, dim] * [q_t, dim, k_t] = [q_t, B*H, k_t] -> [B*H, q_t, k_t]
    rel = torch.matmul(r_q, Rt.transpose(1, 2)).transpose(0, 1)
    # [B*H, q_t, k_t] -> [B, H, q_t, k_t]
    rel = rel.view(B, n_head, t, t)

    attn[:, :, :, :] = (
        attn[:, :, :, :].view(B, -1, t, t, 1, 1)
        + rel[:, :, :, :, None, None]
    ).view(B, -1, t, t)

    return attn



class AdaptivePoolAttention(nn.Module):
    def __init__(
        self,
        input_dim,
        base_dim,
        num_heads,
        qkv_bias=False,
        drop_rate=0.0,
        norm_layer=nn.LayerNorm,
        pooler=nn.AdaptiveAvgPool3d,
        pool_size=(None, 1, 1),
    ):
        super().__init__()
        self.input_dim = input_dim
        self.base_dim = base_dim
        self.num_heads = num_heads
        self.drop_rate = drop_rate
        head_dim = base_dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(input_dim, base_dim * 3, bias=qkv_bias)
        self.norm_q = norm_layer(head_dim)
        self.norm_k = norm_layer(head_dim)
        self.norm_v = norm_layer(head_dim)

        self.proj = nn.Linear(base_dim, base_dim)

        # self.pooler = pooler(pool_size)
        # self.pool_size = pool_size
        self.pooler_q = nn.Conv3d(head_dim, head_dim, kernel_size=(1, 7, 7))
        self.pooler_k = nn.Conv3d(head_dim, head_dim, kernel_size=(1, 7, 7))
        self.pooler_v = nn.Conv3d(head_dim, head_dim, kernel_size=(1, 7, 7))

        if drop_rate > 0.0:
            self.proj_drop = nn.Dropout(drop_rate)

        self.rel_pos_t = nn.Parameter(
            torch.zeros(2 * 64 - 1, head_dim)
        )

    def forward(self, x):
        qkv = self.qkv(x)
        qkv = rearrange(qkv, "b t h w (qkv n c) -> qkv (b n) c t h w", qkv=3, n=self.num_heads)

        q, k, v = qkv[0], qkv[1], qkv[2]
        q, k, v = self.pooler_q(q).flatten(2), self.pooler_k(k).flatten(2), self.pooler_v(v).flatten(2)
        q = rearrange(q, "(b n) c t -> b n t c", b=x.size(0))
        k = rearrange(k, "(b n) c t -> b n t c", b=x.size(0))
        v = rearrange(v, "(b n) c t -> b n t c", b=x.size(0))

        q = self.norm_q(q)
        k = self.norm_k(k)
        v = self.norm_v(v)

        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = cal_rel_pos_temporal(
            attn,
            q,
            self.rel_pos_t,
        )
        attn = attn.softmax(dim=-1)
        x = attn @ v
        x = x + q
        x = rearrange(x, "b n t c -> b t (n c)")

        x = self.proj(x)
        if self.drop_rate > 0.0:
            x = self.proj_drop(x)

        # if self.pool_size[1] != 1 or self.pool_size[2] != 1:
        #     x = rearrange(x, "b (t h w) c -> b t h w c", h=self.pool_size[1], w=self.pool_size[2])

        return x

class AdaptivePooler(nn.Module):
    def __init__(
        self,
        input_dim,
        base_dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        norm_eps=1e-6,
        pooler=nn.AdaptiveAvgPool3d,
        pool_size=(None, 1, 1),
        up_rate=None,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.base_dim = base_dim
        norm_layer = partial(norm_layer, eps=norm_eps)

        # Attention
        self.norm1 = norm_layer(input_dim)
        self.attn = AdaptivePoolAttention(
            input_dim,
            base_dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate,
            norm_layer=norm_layer,
            pooler=pooler,
            pool_size=pool_size,
        )
        self.drop_path = (
            DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        )
        # MLP
        self.norm2 = norm_layer(self.base_dim)
        mlp_hidden_dim = int(base_dim * mlp_ratio)

        if up_rate is not None and up_rate > 1:
            mlp_dim_out = base_dim * up_rate
        else:
            mlp_dim_out = base_dim

        self.mlp = Mlp(
            in_features=base_dim,
            hidden_features=mlp_hidden_dim,
            out_features=mlp_dim_out,
            act_layer=act_layer,
            drop_rate=drop_rate,
        )

        # Skip connection
        if base_dim != mlp_dim_out:
            self.proj = nn.Linear(base_dim, mlp_dim_out)
        else:
            self.proj = nn.Identity()

        # self.pool_skip = pooler(pool_size)
        self.pool_skip = nn.Conv3d(input_dim, base_dim, kernel_size=(1, 7, 7))
        # if input_dim != base_dim:
        #     self.pool_proj = nn.Linear(input_dim, base_dim)
        # else:
        #     self.pool_proj = nn.Identity()
        self.pool_proj = nn.Identity()

        self.proj_norm = norm_layer(mlp_dim_out)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        pool_skip = self.pool_skip(x).flatten(2).transpose(-2, -1)
        x = rearrange(x, "b c ... -> b ... c")
        x = self.drop_path(self.attn(self.norm1(x))) + self.pool_proj(pool_skip)
        x = self.drop_path(self.mlp(self.norm2(x))) + self.proj(x)
        x = self.proj_norm(x)
        x = rearrange(x, "b ... c -> b c ...")
        return x

# class AdaptivePoolingLayer(nn.Module):
#     def __init__(
#         self,
#         input_dim,
#         base_dim,
#         num_heads,
#         mlp_ratio=4.0,
#         qkv_bias=True,
#         drop_rate=0.0,
#         drop_path=0.0,
#         act_layer=nn.GELU,
#         norm_layer=nn.LayerNorm,
#         norm_eps=1e-6,
#         pooler=nn.AdaptiveAvgPool3d,
#         pool_size=(None, 1, 1),
#         up_rate=None,
#     ):
#         super().__init__()

#         self.input_dim = input_dim
#         self.base_dim = base_dim
#         norm_layer = partial(norm_layer, eps=norm_eps)

#         # Attention
#         self.norm1 = norm_layer(input_dim)
#         self.attn = AdaptivePoolAttention(
#             input_dim,
#             base_dim,
#             num_heads=num_heads,
#             qkv_bias=qkv_bias,
#             drop_rate=drop_rate,
#             norm_layer=norm_layer,
#             pooler=pooler,
#             pool_size=pool_size,
#         )
#         self.drop_path = (
#             DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
#         )
#         # MLP
#         self.norm2 = norm_layer(self.base_dim)
#         mlp_hidden_dim = int(base_dim * mlp_ratio)

#         if up_rate is not None and up_rate > 1:
#             mlp_dim_out = base_dim * up_rate
#         else:
#             mlp_dim_out = base_dim

#         self.mlp = Mlp(
#             in_features=base_dim,
#             hidden_features=mlp_hidden_dim,
#             out_features=mlp_dim_out,
#             act_layer=act_layer,
#             drop_rate=drop_rate,
#         )

#         # Skip connection
#         if base_dim != mlp_dim_out:
#             self.proj = nn.Linear(base_dim, mlp_dim_out)
#         else:
#             self.proj = nn.Identity()

#         self.pool_skip = pooler(pool_size)
#         if input_dim != base_dim:
#             self.pool_proj = nn.Linear(input_dim, base_dim)
#         else:
#             self.pool_proj = nn.Identity()

#         self.proj_norm = norm_layer(mlp_dim_out)
#         self.apply(self._init_weights)

#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             nn.init.trunc_normal_(m.weight, std=0.02)
#             if isinstance(m, nn.Linear) and m.bias is not None:
#                 nn.init.zeros_(m.bias)

#         elif isinstance(m, nn.LayerNorm):
#             nn.init.ones_(m.weight)
#             nn.init.zeros_(m.bias)

#     def forward(self, x):
#         pool_skip = self.pool_skip(x).flatten(2).transpose(-2, -1)
#         x = rearrange(x, "b c ... -> b ... c")
#         x = self.drop_path(self.attn(self.norm1(x))) + self.pool_proj(pool_skip)
#         x = self.drop_path(self.mlp(self.norm2(x))) + self.proj(x)
#         x = self.proj_norm(x)
#         x = rearrange(x, "b ... c -> b c ...")
#         return x
# class AdaptivePooler(nn.Module):
#     def __init__(
#         self,
#         input_dim,
#         base_dim,
#         num_heads,
#         mlp_ratio=4.0,
#         qkv_bias=True,
#         drop_rate=0.0,
#         drop_path=0.0,
#         act_layer=nn.GELU,
#         norm_layer=nn.LayerNorm,
#         norm_eps=1e-6,
#         pooler=nn.AdaptiveAvgPool3d,
#         num_layers=1,
#         up_rate=None,
#     ):
#         super().__init__()

#         self.input_dim = input_dim
#         self.base_dim = base_dim
#         norm_layer = partial(norm_layer, eps=norm_eps)

#         self.layers = nn.ModuleList([AdaptivePoolingLayer(
#             input_dim if i == 0 else base_dim,
#             base_dim,
#             num_heads=num_heads,
#             mlp_ratio=mlp_ratio,
#             qkv_bias=qkv_bias,
#             drop_rate=drop_rate,
#             drop_path=drop_path,
#             act_layer=act_layer,
#             norm_layer=norm_layer,
#             pooler=pooler,
#             pool_size=(None, int(2 ** (num_layers - i - 1)), int(2 ** (num_layers - i - 1))),
#             up_rate=up_rate if i + 1 == num_layers else None
#         ) for i in range(num_layers)])

#     def forward(self, x):
#         for layer in self.layers:
#             print(x.size())
#             x = layer(x)
#         return x


if __name__ == '__main__':
    import torch
    model = AdaptivePoolAttention(544, 128)
    x = torch.randn(1, 544, 10, 10, 10)
    print(x.size())
    out = model(x)
    print(out.size())