import sys
sys.path.append(".")

import argparse
import tqdm
import os

from planning.analysis.create_video import generate_video_from_images
from utils.video import merge_video

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str, default="no_git/system/tamp_action_analysis_mid_450_actions_exp_2")
    parser.add_argument('--override', action='store_true')
    parser.add_argument('--merge_videos', action='store_true')
    args = parser.parse_args()
    experiments = [os.path.join(args.folder, experiment) for experiment in os.listdir(args.folder)]
    experiments.sort()
    
    override = args.override if args.override is not None else False
    videos = []
    for action in ["Pick", "Place"]:
        for experiment in tqdm.tqdm(experiments): #'no_git/system/tamp_action_analysis_mid_450_actions_exp_2/28_single'
            experiment_instance = os.path.join(experiment, os.listdir(experiment)[0], "actions") #'no_git/system/tamp_action_analysis_mid_450_actions_exp_2/28_single/11_8_9_25_46_158880/actions'
            action_experiment_path = os.path.join(experiment_instance, action) #'no_git/system/tamp_action_analysis_mid_450_actions_exp_2/28_single/11_8_9_25_46_158880/actions/Pick'
            if not os.path.exists(action_experiment_path):
                continue
            for action_instance in os.listdir(action_experiment_path):
                action_experiment_instance_path = os.path.join(action_experiment_path, action_instance) #'no_git/system/tamp_action_analysis_mid_450_actions_exp_2/28_single/11_8_9_25_46_158880/actions/Pick/7'
                generate_video_from_images(
                    folder_path=action_experiment_instance_path,
                    video_name="all_tries_video.avi",
                    sampling_rate=4,
                    override=override
                )
                videos.append(os.path.join(action_experiment_instance_path, "all_tries_video.avi"))
 
    #merge all videos
    if args.merge_videos:
        merge_video(
            videos_paths=videos,
            output_path="merged_video.mp4"
        )