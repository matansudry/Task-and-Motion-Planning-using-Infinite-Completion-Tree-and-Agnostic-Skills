import sys
sys.path.append(".")

import os
from deepdiff import DeepDiff
import pybullet as p
import argparse
from utils.general_utils import load_pickle, save_pickle
from utils.config_utils import load_cfg
from utils.pddl_dataclass import PDDLConfig
from stap.envs.pybullet.utils import RedirectStream
from generate_data import GenerateDataset
with RedirectStream(sys.stderr):
    import pybullet as p
import numpy as np
import cv2
from lightning_fabric.utilities.seed import seed_everything
from ctrlutils import eigen
from stap.envs.pybullet.table_env import Task

def connect_pybullet(gui: bool = True, gui_kwargs = {}) -> int:
    if not gui:
        with RedirectStream():
            physics_id = p.connect(p.DIRECT, options=gui_kwargs["options"])
    elif not os.environ["DISPLAY"]:
        raise p.error
    else:
        with RedirectStream():
            physics_id = p.connect(p.GUI, options=gui_kwargs["options"])

        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=physics_id)
        p.configureDebugVisualizer(
            p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW,
            0,
            physicsClientId=physics_id,
        )
        p.configureDebugVisualizer(
            p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0, physicsClientId=physics_id
        )
        p.configureDebugVisualizer(
            p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0, physicsClientId=physics_id
        )
        p.configureDebugVisualizer(
            p.COV_ENABLE_SHADOWS,
            gui_kwargs.get("shadows", 0),
            physicsClientId=physics_id,
        )
        p.resetDebugVisualizerCamera(
            cameraDistance=0.25,
            cameraYaw=90,
            cameraPitch=-48,
            cameraTargetPosition=[0.76, 0.07, 0.37],
            physicsClientId=physics_id,
        )
    p.setTimeStep(1 / 60)
    return physics_id

DEFAULT_OPTIONS = {
    "background_color_red": 0.12,
    "background_color_green": 0.12,
    "background_color_blue": 0.25,
}

def to_str_kwarg(kv) -> str:
    return f"--{kv[0]}={kv[1]}"

def show_image(image:np.array):
    """_summary_

    Args:
        image (np.array): _description_
    """
    cv2.imshow("image", image)
    cv2.waitKey(0)

def get_env(args):
    
    cfg = load_cfg(config_path=args.config_path, load_as_edict=True)
    cfg.general_params.output_path = os.path.join(args.output_path)
    cfg.general_params.seed = args.seed
    if args.primitive is not None:
        cfg.env.primitive = args.primitive
    """
    from utils.config_utils import save_cfg
    save_cfg(
        cfg=cfg,
        output_folder=cfg.general_params.output_path
    )
    """
    cfg.env.pddl_config = PDDLConfig()
    
    data_generator = GenerateDataset(
        cfg=cfg
    )
    
    import tempfile
    data_generator.tmpdir = "no_git/tmp/"
    
    return data_generator, cfg

def validate_path(sample_path:str, cfg:dict):
    primitive = cfg['env']['primitive']
    primitive = primitive[0].upper() + primitive[1:]
    assert primitive == sample_path.split("/")[-3]

def reload_action():
    cnt = 0
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default="configs/generate_dataset_config_playground.yml") #"configs/generate_dataset_config.yml") #
    parser.add_argument('--output_path', type=str, default="no_git/playground/samples") #"no_git/dataset")
    parser.add_argument('--primitive', type=str, default=None,\
        choices=["pick", "place", "pull", "push"])
    parser.add_argument('--seed', type=int, default=700)
    parser.add_argument('--number_of_samples', type=int, default=10)
    
    
    args = parser.parse_args()
    for _ in range(10):
        path = "no_git/playground/samples/Pick/Place/1/sample.pickle"
        sample = load_pickle(path)

        data_generator, cfg = get_env(args)
        validate_path(sample_path=path, cfg=cfg)
        data_generator.reset_env(task=sample['task'])
        #data_generator.env.reset()
        data_generator.set_state(
            observation=sample['start_observation'],
            robot_state=sample['robot_start_state'],
            state_path=sample['start_state_path']
            )
        image = data_generator.env.render()
        #show_image(image=image)
        #p.restoreState(fileName=sample['start_state_path'])
        #new_state_path, observation, state = data_generator.get_state()
        
        #for _ in range(20):
        #    p.stepSimulation()
        
        data_generator.env.record_start()
        _, reward, _, _, info = data_generator.step(action=sample['action'])
        data_generator.env.record_stop()
        data_generator.env.record_save(
            path = "no_git/tmp/video_"+str(cnt)+".gif"
        )
        cnt+=1
        p.disconnect(data_generator.env.physics_id) 

        del data_generator

    temp=1

if __name__ == "__main__":
    reload_action()
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default="configs/generate_dataset_config_playground.yml") #"configs/generate_dataset_config.yml") #
    parser.add_argument('--output_path', type=str, default="no_git/playground/samples") #"no_git/dataset")
    parser.add_argument('--primitive', type=str, default=None,\
        choices=["pick", "place", "pull", "push"])
    parser.add_argument('--seed', type=int, default=700)
    parser.add_argument('--number_of_samples', type=int, default=10)
    
    
    args = parser.parse_args()
    seed_everything(seed=2)
    sample = load_pickle("no_git/playground/samples/Pick/1/sample.pickle") #"no_git/playground/samples/Pick/Place/1/sample.pickle")
    data_generator = get_env(args)
    data_generator.reset_env(task=sample['task'])
    data_generator.set_state(
        observation=sample['end_observation'],
        robot_state=sample['robot_end_state'],
        state_path=sample['end_state_path']
        )
    image = data_generator.env.render()
    show_image(image=image)
    temp=1