import json
import os
import shutil

import numpy as np


def gen_activitynet_frames_info(src_folder, tgt_folder, target_frames, anno_path):
    with open(anno_path) as f:
        anno_dict = json.load(f)['database']

    result_dict = {}
    for vid in anno_dict.keys():
        if anno_dict[vid]['subset'] == "testing":
            continue

        src_vid_folder = os.path.join(src_folder, 'v_' + vid)
        frame_names = os.listdir(src_vid_folder)
        num_frames = len(frame_names)
        if num_frames == 0:
            print('v_' + vid, "folder is empty!")

        tgt_vid_folder = os.path.join(tgt_folder, vid)
        os.makedirs(tgt_vid_folder, exist_ok=True)
        indices = np.linspace(0, 1, target_frames)
        indices = np.round(indices * (num_frames - 1))

        for tgt_idx, src_idx in enumerate(indices):
            src_path = os.path.join(src_folder, f"img_{int(src_idx + 1):07d}.jpg")
            tgt_path = os.path.join(tgt_folder, f"img_{tgt_idx + 1:07d}.jpg")
            shutil.copy2(src_path, tgt_path)

        duration = anno_dict[vid]['duration']
        feature_fps = target_frames / duration
        result_dict[vid] = {'feature_length': num_frames, 'feature_second': duration, 'feature_fps': feature_fps}

    with open('data/activitynet/activitynet_{}_info.json'.format(target_frames), 'w') as f:
        json.dump(result_dict, f)

if __name__ == '__main__':
    anno_path = 'data/activitynet/activity_net.v1-3.min.json'
    src_dir = 'data/activitynet/activitynet_384frames'
    extract_frames = 384
    # extract_frames = 768
    tgt_dir = 'data/videos/activitynet/'

    gen_activitynet_frames_info(src_dir, tgt_dir, extract_frames, anno_path)