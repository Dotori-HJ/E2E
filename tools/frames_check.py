import json
import os

if __name__ == '__main__':
    anno_path = 'data/activitynet/activity_net.v1-3.min.json'
    with open(anno_path) as f:
        anno_dict = json.load(f)['database']

    # src_folder = 'data/activitynet/384frames'
    src_folder = 'data/frames/activitynet'
    for vid in anno_dict.keys():
        if anno_dict[vid]['subset'] == "testing":
            continue
        src_vid_folder = os.path.join(src_folder, 'v_' + vid)
        frame_names = os.listdir(src_vid_folder)
        num_frames = len(frame_names)
        print(vid, num_frames)