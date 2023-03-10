import json
import os
import os.path as osp
import shutil
from shutil import SameFileError

import cv2
import numpy as np


def gen_activitynet_frames_info(frame_dir, video_paths, target_frames, anno_path):
    # files = os.listdir(video_paths)
    with open(anno_path) as f:
        anno_dict = json.load(f)['database']

    num = 0
    result_dict = {}
    for vid in anno_dict.keys():
        if anno_dict[vid]['subset'] == "testing":
            continue

        folder_path = os.path.join(frame_dir, vid)
        frame_names = os.listdir(folder_path)
        num_frames = len(frame_names)

        if num_frames < target_frames:
            backup_path = folder_path + ".bak"
            if not os.path.exists(backup_path):
                shutil.copytree(folder_path, backup_path)
            indices = np.linspace(0, 1, target_frames)
            indices = np.round(indices * (num_frames - 1))

            for tgt_idx, src_idx in enumerate(indices):
                src_path = os.path.join(backup_path, f"img_{int(src_idx + 1):07d}.jpg")
                tgt_path = os.path.join(folder_path, f"img_{tgt_idx + 1:07d}.jpg")
                shutil.copy2(src_path, tgt_path)
                # try:
                #     shutil.copy2(src_path, tgt_path)
                # except SameFileError:
                #     pass
        # if frames != num_frames:
        #     print(folder_path)
        # num_frames = len(os.listdir(osp.join(video_paths, vid)))
        # feature_second = num_frames / anno_dict
        # video_second = anno_dict[vid]['duration']
        # diff = abs(feature_second - video_second)
        # if diff > 3:
        #     print(vid, feature_second, video_second)
        # feature_fps = anno_dict[vid]['fps'] / 8
        diff = target_frames - num_frames
        fps = num_frames / anno_dict[vid]['duration']
        duration = anno_dict[vid]['duration']
        # if diff > 0:
        #     duration = anno_dict[vid]['duration'] + (1 / fps * diff)
        #     num += 1
        #     print(folder_path)
        # else:
        #     duration = anno_dict[vid]['duration']
        feature_fps = target_frames / duration
        result_dict[vid] = {'feature_length': num_frames, 'feature_second': duration, 'feature_fps': feature_fps}
        # result_dict[vid] = {'feature_length': frames, 'feature_second': 384 * anno_dict[vid]['duration'] / num_frames, 'feature_fps': feature_fps}

    print(num)
    # result_dict['num_frames'] = num_frames
    # if not osp.exists('data/thumos14'):
    #     os.makedirs('data/thumos14')

    with open('data/activitynet/activitynet_{}_info.json'.format(target_frames), 'w') as f:
        json.dump(result_dict, f)


def convert_gt(anno_path, target_path):
    new_dict = dict()
    with open(anno_path, 'rt') as f:
        anno_dict = json.load(f)

    new_dict['taxonomy'] = anno_dict['taxonomy']
    new_dict['version'] = anno_dict['version']
    new_dict['database'] = dict()

    for name in anno_dict['database']:
        new_dict['database']['v_' + name] = anno_dict['database'][name]

    with open(target_path, 'wt') as f:
        json.dump(new_dict, f)

if __name__ == '__main__':
    # frame_dir = 'data/thumos14/thumos14_img15fps'
    # gen_thumos14_frames_info(frame_dir, 15)
    anno_path = 'data/activitynet/activity_net.v1-3.min.json'
    target_path = 'data/activitynet/gt.json'
    frame_dir = 'data/activitynet/activitynet_768_frames'
    # convert_gt(anno_path, target_path)
    # exit()

    video_dirs = [
        '/home/ds/SSD2/Videos/ActivityNet/archives/v1-3/train_val',
        '/home/ds/SSD2/Videos/ActivityNet/archives/v1-2/train',
        '/home/ds/SSD2/Videos/ActivityNet/archives/v1-2/val'
    ]
    video_paths = []
    # for video_dir in video_dirs:
    #     video_names = os.listdir(video_dir)
    #     for video_name in video_names:
    #         video_paths.append(os.path.join(video_dir, video_name))

    gen_activitynet_frames_info(frame_dir, video_paths, 768, target_path)