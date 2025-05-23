import cv2
import os
from moviepy.editor import VideoFileClip, concatenate_videoclips
from PIL import Image
import numpy as np

def save_image(image:np.ndarray, path:str):
    img = Image.fromarray(image, 'RGB')
    img.save(path)

def generate_video(images:list, output_path:str, video_name:str, sampling_rate:int=1):
    path = os.path.join(output_path, video_name)
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"DIVX"), sampling_rate, tuple([images[0].shape[1], images[0].shape[0]]))
    for i in range(len(images)):
        out.write(images[i])
    out.write(images[-1])
    out.release()

def merge_video(videos_paths:list, output_path:str):
    assert ".mp4" in output_path, f'mp4 is missing in {output_path}'
    loaded_video_list = []

    for video_path in videos_paths:
        print(f"Adding video file:{video_path}")
        loaded_video_list.append(VideoFileClip(video_path))

    final_clip = concatenate_videoclips(loaded_video_list)

    final_clip.write_videofile(output_path)