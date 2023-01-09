import os

import cv2
import numpy as np
from tqdm import tqdm

src_folder = "/home/ds/HDD2/ActivityNet/archives/v1-3/train_val"
video_files = os.listdir(src_folder)

num_samples = 384

dst_folder = "/home/ds/HDD2/ActivityNet_frames/v1-3/train_val"


def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)
    elif width == height:
        dim = (width, height)
    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized


for video_file in tqdm(video_files):
    src_path = os.path.join(src_folder, video_file)
    video_name = os.path.splitext(video_file)[0]

    target_folder = os.path.join(dst_folder, video_name)
    os.makedirs(target_folder, exist_ok=True)

    cap = cv2.VideoCapture(src_path)

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    indices = np.linspace(0, num_frames-1, num=num_samples)
    indices = np.floor(indices)
    indices = indices.tolist()

    resize_args = {}
    if width > height:
        resize_args['width'] = None
        resize_args['height'] = 256
        if height > 256:
            resize_args['inter'] = cv2.INTER_AREA
        else:
            resize_args['inter'] = cv2.INTER_CUBIC
    elif width == height:
        resize_args['width'] = 256
        resize_args['height'] = 256
        if height > 256:
            resize_args['inter'] = cv2.INTER_AREA
        else:
            resize_args['inter'] = cv2.INTER_CUBIC
    else:
        resize_args['width'] = 256
        resize_args['height'] = None
        if width > 256:
            resize_args['inter'] = cv2.INTER_AREA
        else:
            resize_args['inter'] = cv2.INTER_CUBIC

    i = 0
    file_num = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if file_num == num_samples:
            break

        if indices[0] == i:
            resized_frame = image_resize(frame, **resize_args)
            try:
                while indices[0] == i:
                    file_num += 1
                    img_path = os.path.join(target_folder, f"img_{file_num:07d}.jpg")
                    cv2.imwrite(img_path, resized_frame)
                    indices.pop(0)
            except IndexError as e:
                if file_num == num_samples:
                    break
                else:
                    assert True, "Error"

        i += 1

    cap.release()
