from timm.data import create_transform

transform = create_transform(
    # input_size=384,
    input_size=224,
    is_training=True,
    # color_jitter=config.AUG.COLOR_JITTER if config.AUG.COLOR_JITTER > 0 else None,
    # auto_augment=config.AUG.AUTO_AUGMENT if config.AUG.AUTO_AUGMENT != 'none' else None,
    # re_prob=config.AUG.REPROB,
    # re_mode=config.AUG.REMODE,
    # re_count=config.AUG.RECOUNT,
    # interpolation=config.DATA.INTERPOLATION,
)
#     def __call__(self, x):
#         ops = np.random.choice(
#             self.ops,
#             self.N,
#             # replace=self.choice_weights is None,
#             # p=self.choice_weights,
#         )
#         for op in ops:
#             x = torch.stack([op(_) for _ in x])
#         return x
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from einops import rearrange, repeat
from PIL import Image

from datasets.utils import spatial_sampling, tensor_normalize
from video_transforms import create_random_augment

# print(transform)

# # from torchvision.transforms._transforms_video import 
# from PIL import Image

# import random

# import kornia as K
# import kornia.augmentation as KA
# import kornia.geometry as KG
# import numpy as np

# from rand_augment import rand_augment_transform

# KG.translate()
# class RandAugment:
#     _MAX_LEVEL = 10
#     def __init__(self, N, M):
#         self.N = N
#         self.M = M
#         self.ratio = M / self._MAX_LEVEL
#         self.ops = [
#             KA.RandomRotation(self.ratio * 30, same_on_batch=True),
#             KA.RandomAffine(degrees=0, shear=(0, self.ratio * 0.3, 0, 0), same_on_batch=True),    # Shear X
#             KA.RandomAffine(degrees=0, shear=(0, 0, 0, 0.3),same_on_batch=True),    # Shear Y
#             KA.RandomAffine(degrees=0, translate=(0.1, 0), same_on_batch=True),    # TranslateX
#             KA.RandomAffine(degrees=0, translate=(0, 0.1), same_on_batch=True),    # TranslateY
#             # KA.Random,    # Color
#             KA.RandomSharpness(0.5, same_on_batch=True),
#             KA.RandomSolarize(same_on_batch=True),
#             K.enhance.equalize,
#             # KA.
#             KA.RandomHorizontalFlip(p=0.5, same_on_batch=True),
#         ]
#         print(self.ops)
#         KA.RandomSolarize(same_on_batch=True),





# img = cv2.imread("000000.png")
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# img = np.transpose(img, [2, 0, 1]).astype(np.float32)
# img /= 255.
# img = torch.from_numpy(img).contiguous().to("cuda")


# aa_params = {"translate_const": int(224 * 0.45)}


# aa_params["interpolation"] = Image.BILINEAR
# transform = rand_augment_transform("rand-m7-n4-mstd0.5-inc1", aa_params)


# class GPUAugment:
#     def __init__(self, transforms):
#         self.transforms = transforms

#     def __call__(self, x):
#         for op in self.transforms:
#             x = torch.stack([op(_) for _ in x])
#         return x

# transform = RandAugment(2, 7)
# import kornia.augmentation as KA

# gpu_transforms = GPUAugment([
#     KA.RandomAffine(30, translate=0.1, shear=0.3, p=0.5, same_on_batch=True),
#     KA.ColorJiggle(0.1, 0.1, 0.1, 0.1, p=0.5, same_on_batch=True),
#     KA.RandomHorizontalFlip(p=0.5, same_on_batch=True),
#     # KE.Normalize(mean=mean / 255, std=std / 255),
# ])


img = Image.open("000000.png")
buffer = [img for i in range(16)]
# AugmentOp()
# img = repeat(img, "c h w -> b t c h w", b=4, t=8)
print(buffer[0].size)
transform = create_random_augment(buffer[0].size, "rand-m7-n4-mstd0.5-inc1", "bilinear")
buffer = transform(buffer)

buffer = [transforms.ToTensor()(img) for img in buffer]
buffer = torch.stack(buffer) # T C H W
print(buffer.size(), buffer[0].size)
exit()

buffer = buffer.permute(0, 2, 3, 1) # T H W C

# T H W C
buffer = tensor_normalize(
    buffer, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
)
# T H W C -> C T H W.
buffer = buffer.permute(3, 0, 1, 2)
# Perform data augmentation.
scl, asp = (
    [0.08, 1.0],
    [0.75, 1.3333],
)
buffer = spatial_sampling(
    buffer,
    spatial_idx=-1,
    min_scale=256,
    max_scale=320,
    crop_size=224,
    random_horizontal_flip=True,
    inverse_uniform_sampling=False,
    aspect_ratio=asp,
    scale=scl,
    motion_shift=False
)

# if self.rand_erase:
#     erase_transform = RandomErasing(
#         args.reprob,
#         mode=args.remode,
#         max_count=args.recount,
#         num_splits=args.recount,
#         device="cpu",
#     )
#     buffer = buffer.permute(1, 0, 2, 3)
#     buffer = erase_transform(buffer)
#     buffer = buffer.permute(1, 0, 2, 3)

from torchvision.utils import save_image

# for i in range(4):
    # save_image(buffer[i], f"samples/sample_{i}.png")
print(buffer.size())
# buffer = rearrange(buffer, "b t c h w -> (b t) c h w")
save_image(buffer.permute(1, 0, 2, 3).cpu(), "samples/sample.png", nrow=8)
