import sys
sys.path.append(".")

import argparse
import torch
from lightning_fabric.utilities.seed import seed_everything
import tqdm

from utils.config_utils import load_cfg
from p_estimator.data.data_module import PEstimatorDataModule 
from p_estimator.network.attention import AttentaionNetwork

def main(cfg:dict):
    data_module = PEstimatorDataModule(
        cfg=cfg.DATA,
        primitive=cfg.GENERAL_PARAMS.primitive
    )
    data_module.setup(stage="fit")
    data_loader = data_module.train_dataloader()
    network = AttentaionNetwork().to("cuda")


    for i, data in tqdm.tqdm(enumerate(data_loader)):
        data['state'] = data['state'].to("cuda")
        output = network(data)

if __name__ == '__main__':
    #os.environ['CUDA_VISIBLE_DEVICES'] = "1"
    seed_everything(seed=0)

    assert torch.cuda.is_available()

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default="p_estimator/config/train_config.yml")
    args = parser.parse_args()

    cfg = load_cfg(config_path=args.config_path, load_as_edict=True)


    main(cfg=cfg)