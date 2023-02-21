import argparse
import concurrent.futures
import json
import os
import os.path as osp
from re import sub

import cv2
import numpy as np


def extract_frames(video_path, dst_dir, fps):
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    video_fname = osp.basename(video_path)

    cap = cv2.VideoCapture(video_fname)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    # fps = cap.get(cv2.CAP_PROP_FPS)
    # frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # # Calculate duration
    # duration = frame_count / fps

    # if not osp.exists(video_path):
    #     subdir = 'test_set/TH14_test_set_mp4' if 'test' in video_fname else 'Validation_set/videos'
    #     url = f'https://crcv.ucf.edu/THUMOS14/{subdir}/{video_fname}'
    #     os.system('wget {} -O {} --no-check-certificate'.format(url, video_path))
    # cmd = 'ffmpeg -i "{}"  -filter:v "fps=fps={}" "{}/img_%07d.jpg"'.format(video_path, fps, dst_dir)
    if width > height:
        # cmd = f'ffmpeg -i "{video_path}" -vf "scale=256:-1, setpts=N/TB" -vframes 384 -r 1 "{dst_dir}/img_%07d.jpg"'    # ActivityNet v1.3
        cmd = f'ffmpeg -i "{video_path}" -vf "scale=256:-1" "{dst_dir}/img_%07d.jpg"'    # ActivityNet v1.3
        # cmd = f'ffmpeg -i "{video_path}" -vf "scale=256:-1, setpts=N/((FRAME_RATE)*TB)" -r 1 -vframes 384 "{dst_dir}/img_%07d.jpg"'    # ActivityNet v1.3
        # cmd = f'ffmpeg -i "{video_path}" -vf "scale=256:-1, setpts=N/TB" -vframes 384 "{dst_dir}/img_%07d.jpg"'    # ActivityNet v1.3
    else:
        # cmd = f'ffmpeg -i "{video_path}" -vf "scale=-1:256, setpts=N/TB" -vframes 384 -r 1 "{dst_dir}/img_%07d.jpg"'    # ActivityNet v1.3
        cmd = f'ffmpeg -i "{video_path}" -vf "scale=-1:256" "{dst_dir}/img_%07d.jpg"'    # ActivityNet v1.3
        # cmd = f'ffmpeg -i "{video_path}" -vf "scale=-1:256, setpts=N/((FRAME_RATE)*TB)" -r 1 -vframes 384 "{dst_dir}/img_%07d.jpg"'    # ActivityNet v1.3
        # cmd = f'ffmpeg -i "{video_path}" -vf "scale=-1:256, setpts=N/TB" -vframes 384 "{dst_dir}/img_%07d.jpg"'    # ActivityNet v1.3

    print(cmd)
    ret_code = os.system(cmd)
    if ret_code == 0:
        os.system('touch logs/frame_extracted_{}fps/{}'.format(fps, osp.splitext(osp.basename(video_path))[0]))
    return ret_code == 0


def parse_args():
    parser = argparse.ArgumentParser('Extract frames')
    parser.add_argument('--video_dir', help='path to the parent dir of video directory')
    parser.add_argument('--frame_dir', help='path to save extracted video frames')
    parser.add_argument('--fps', type=int, default=10)
    parser.add_argument('-s', '--start', type=int)
    parser.add_argument('-e', '--end', type=int)

    args = parser.parse_args()
    return args



def mkdir_if_missing(dirname):
    if not osp.exists(dirname):
        os.makedirs(dirname)

def main(subset):
    args = parse_args()

    log_dir = 'logs/frame_extracted_{}fps'.format(args.fps)
    mkdir_if_missing(log_dir)
    # mkdir_if_missing(args.video_dir)

    # database = json.load(open('data/thumos14/th14_annotations_with_fps_duration.json'))['database']
    # database = json.load(open('/home/ds/HDD2/ActivityNet/activity_net.v1-3.min.json'))['database']
    # vid_names = list(sorted([x for x in database if database[x]['subset'] == subset]))

    vid_names = os.listdir(args.video_dir)
    # vid_names = []

    start_ind = 0 if args.start is None else args.start
    end_ind = len(vid_names) if args.end is None else min(args.end, len(vid_names))

    # vid_names = vid_names[args.start:args.end]

    finished = os.listdir('logs/frame_extracted_{}fps'.format(args.fps))
    videos_todo = list(sorted(set(vid_names).difference(finished)))
    with concurrent.futures.ProcessPoolExecutor(256) as f:
        # futures = [f.submit(extract_frames, osp.join(args.video_dir, 'v_' + x + '.mp4'),
        futures = [f.submit(extract_frames, osp.join(args.video_dir, x),
                            osp.join(args.frame_dir, os.path.splitext(x)[0]), args.fps) for x in videos_todo]

    for f in futures:
        f.result()


if __name__ == '__main__':
    # main('val')
    # main('test')
    main('train')
    main('validation')

# thumos14
# python tools/extract_frames.py --video_dir data/thumos14/videos --frame_dir data/thumos14/img10fps --fps  10 -e 4

# python tools/extract_frames.py --video_dir /home/ds/HDD2/ActivityNet/archives/v1-3/train_val --frame_dir /home/ds/SSD2/ActivityNet_v1-3_384frames
# python tools/extract_frames.py --video_dir /home/ds/HDD2/ActivityNet/archives/v1-3/train_val --frame_dir /home/ds/SSD2/ActivityNet_v1-3_384frames