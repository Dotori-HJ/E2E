import torch
import torch.nn as nn
from einops import rearrange, reduce

from .pooler import DropPath, Mlp


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

        self.pooler = pooler(pool_size)

        if drop_rate > 0.0:
            self.proj_drop = nn.Dropout(drop_rate)

    def forward(self, x):
        qkv = self.qkv(x)
        qkv = rearrange(qkv, "b t h w (qkv n c) -> qkv (b n) c t h w", qkv=3, n=self.num_heads)

        q, k, v = qkv[0], qkv[1], qkv[2]
        q, k, v = self.pooler(q).flatten(2), self.pooler(k).flatten(2), self.pooler(v).flatten(2)
        q = rearrange(q, "(b n) c t -> b n t c", b=x.size(0))
        k = rearrange(k, "(b n) c t -> b n t c", b=x.size(0))
        v = rearrange(v, "(b n) c t -> b n t c", b=x.size(0))

        q = self.norm_q(q)
        k = self.norm_k(k)
        v = self.norm_v(v)

        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        x = attn @ v
        x = x + q
        x = rearrange(x, "b n t c -> b t (n c)")

        x = self.proj(x)
        if self.drop_rate > 0.0:
            x = self.proj_drop(x)

        return x


class AdaptivePooler(nn.Module):
    def __init__(
        self,
        input_dim,
        base_dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop_rate=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        pooler=nn.AdaptiveAvgPool3d,
        pool_size=(None, 1, 1),
        up_rate=None,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.base_dim = base_dim

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

        self.pool_skip = pooler(pool_size)
        if input_dim != base_dim:
            self.pool_proj = nn.Linear(input_dim, base_dim)
        else:
            self.pool_proj = nn.Identity()

    def forward(self, x):
        pool_skip = self.pool_skip(x).flatten(2).transpose(-2, -1)
        x = rearrange(x, "b c ... -> b ... c")
        x = self.drop_path(self.attn(self.norm1(x))) + self.pool_proj(pool_skip)
        x = self.drop_path(self.mlp(self.norm2(x))) + self.proj(x)
        x = rearrange(x, "b ... c -> b c ...")
        return x

if __name__ == '__main__':
    import torch
    model = AdaptivePoolAttention(544, 128)
    x = torch.randn(1, 544, 10, 10, 10)
    print(x.size())
    out = model(x)
    print(out.size())