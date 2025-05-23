import sys
sys.path.append(".")

import os
import cv2
import numpy as np

from utils.video import generate_video#(images:list, output_path:str, video_name:str, sampling_rate:int=1)

def resize(image:np.array, target_shape:tuple):

    new_image = cv2.resize(image, target_shape[:2], 
                interpolation = cv2.INTER_LINEAR)
    return new_image

def generate_video_from_images(folder_path:str, video_name:str, sampling_rate:int, override:bool=False):
    images = []
    images_names = os.listdir(folder_path)
    images_names.sort()
    if 'all_tries_video.avi' in images_names:
        if override:
            images_names.remove("all_tries_video.avi")
        else:
            return 
    target_shape = cv2.imread(os.path.join(folder_path, images_names[0]), cv2.IMREAD_UNCHANGED).shape
    target_shape = [target_shape[1]*2, target_shape[0]]

    #load start image
    start_image = cv2.imread("assets/start.png", cv2.IMREAD_UNCHANGED)
    start_image = resize(
        image=start_image,
        target_shape=target_shape
    )
    images.append(start_image)

    #load init image
    #init_title_image = cv2.imread("assets/init.png", cv2.IMREAD_UNCHANGED)
    #init_title_image = resize(
    #    image=init_title_image,
    #    target_shape=target_shape
    #)
    #images.append(init_title_image)

    init_image = cv2.imread(os.path.join(folder_path, 'start_image.png'), cv2.IMREAD_UNCHANGED)
    #for _ in range(sampling_rate):
    #    images.append(init_image)

    #load fail image
    success_images = []
    fail_images = []
    for image in images_names:
        if "fail" in image:
            fail_images.append(os.path.join(folder_path, image))
        elif "success" in image:
            success_images.append(os.path.join(folder_path, image))

    #add fail images
    fail_title_image = cv2.imread("assets/fail.png", cv2.IMREAD_UNCHANGED)
    fail_title_image = resize(
        image=fail_title_image,
        target_shape=target_shape
    )
    images.append(fail_title_image)

    for fail_image_path in fail_images:
        fail_image = cv2.imread(fail_image_path, cv2.IMREAD_UNCHANGED)
        fail_image = np.concatenate((init_image, fail_image), axis=1)
        images.append(fail_image)
    
    #add success images
    success_title_image = cv2.imread("assets/success.png", cv2.IMREAD_UNCHANGED)
    success_title_image = resize(
        image=success_title_image,
        target_shape=target_shape
    )
    images.append(success_title_image)

    for success_image_path in success_images:
        success_image = cv2.imread(success_image_path, cv2.IMREAD_UNCHANGED)
        success_image = np.concatenate((init_image, success_image), axis=1)
        images.append(success_image)

    
    #check images
    test_shape = images[0].shape
    for image in images:
        assert image.shape == test_shape

    generate_video(
        images=images,
        output_path=folder_path,
        video_name=video_name,
        sampling_rate=sampling_rate,
    )

if __name__ == "__main__":
    folder_path = "no_git/system/tamp_action_analysis_mid_450_actions_exp_2/0_single/11_8_9_16_24_216324/actions/Pick/1"
    sampling_rate = 1
    video_name = "all_tries_video.avi"
    generate_video_from_images(
        folder_path=folder_path,
        video_name=video_name,
        sampling_rate=sampling_rate,
        override=True
    )