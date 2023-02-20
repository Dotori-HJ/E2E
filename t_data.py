from easydict import EasyDict
from tqdm import tqdm

from datasets import build_dataset
from util.segment_ops import segment_cw_to_t1t2

cfg = EasyDict(
    dataset_name='activitynet',
    feature='img',
    input_type='image',
    encoder='slowfast',
    img_resize=112,
    img_crop_size=96,
    slice_len=384,
    resize_keep_asr=True,
    online_slice=False,
    slice_overlap=0,
    test_slice_overlap=0,
    binary=True,
    rand_augment_param='rand-m7-n4-mstd0.5-inc1',
    rand_erase=False,
    fix_transform=False,
)

# dataset_val, _ = build_dataset(subset='val', args=cfg, mode='val')
# for data, anno in tqdm(dataset_val):
#     if (anno['segments']  * anno['video_duration'] > anno['video_duration']).sum() > 0:
#         pass
#     elif (anno['segments'] < 0).sum() > 0:
#         pass
#     else:
#         continue
#     print(f'video_id: {anno["video_id"]}, {anno}')
dataset_train, gpu_transforms = build_dataset(subset='train', args=cfg, mode='train')
for data, anno in tqdm(dataset_train):
    segments = segment_cw_to_t1t2(anno['segments'])
    if (segments  * anno['video_duration'] > anno['video_duration']).sum() > 0:
        pass
    elif (segments < 0).sum() > 0:
        pass
    else:
        continue
    print(f'video_id: {anno["video_id"]}, {anno}')