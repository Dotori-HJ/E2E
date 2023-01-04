import torch
import torch.optim as optim
from easydict import EasyDict

from models.video_encoder_archs.video_swin import SwinTransformer3D

x = torch.randn(4, 3, 128, 224, 224).cuda()

tiny = EasyDict(
    patch_size=(4,4,4),
    embed_dim=96,
    depths=[2, 2, 6, 2],
    num_heads=[3, 6, 12, 24],
    window_size=(8,7,7),
    mlp_ratio=4.,
    qkv_bias=True,
    qk_scale=None,
    drop_rate=0.,
    attn_drop_rate=0.,
    drop_path_rate=0.2,
    patch_norm=True,
)
small = EasyDict(
    patch_size=(4,4,4),
    embed_dim=96,
    depths=[2, 2, 18, 2],
    num_heads=[3, 6, 12, 24],
    window_size=(8,7,7),
    mlp_ratio=4.,
    qkv_bias=True,
    qk_scale=None,
    drop_rate=0.,
    attn_drop_rate=0.,
    drop_path_rate=0.2,
    patch_norm=True,
)
base = EasyDict(
    patch_size=(4,4,4),
    embed_dim=128,
    depths=[2, 2, 18, 2],
    num_heads=[4, 8, 16, 32],
    window_size=(8,7,7),
    mlp_ratio=4.,
    qkv_bias=True,
    qk_scale=None,
    drop_rate=0.,
    attn_drop_rate=0.,
    drop_path_rate=0.2,
    patch_norm=True,
)


model = SwinTransformer3D(**tiny).cuda()
# model = SwinTransformer3D(**small).cuda()
# model = SwinTransformer3D(**base).cuda()
for param in model.parameters():
    param.requires_grad_(False)
model.eval()
import torch.nn as nn
import torch.nn.functional as F

# cls = nn.Linear(1024, 10).cuda()
# optimizer = optim.Adam(list(model.parameters()) + list(cls.parameters()), lr=1e-4)

# optimizer.zero_grad()

outs = model(x)
print([o.size() for o in outs])
# out = F.adaptive_avg_pool3d(out, (1, 1, 1)).flatten(1)
# out = cls(out)
# loss = out.mean()
# loss.backward()

# optimizer.step()