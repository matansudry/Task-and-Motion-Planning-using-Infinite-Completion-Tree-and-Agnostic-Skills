#!/usr/bin/env python3

import argparse
import pathlib
from pprint import pprint
from typing import Any, Dict, Optional, Sequence, Union

from stap import encoders, envs, trainers
from stap.utils import configs, random


def collect(
    path: Union[str, pathlib.Path],
    policy_checkpoints: Sequence[Union[str, pathlib.Path]],
    trainer_config: Optional[Union[str, pathlib.Path, Dict[str, Any]]] = None,
    env_config: Optional[Union[str, pathlib.Path, Dict[str, Any]]] = None,
    encoder_config: Optional[Union[str, pathlib.Path, Dict[str, Any]]] = None,
    resume: bool = False,
    overwrite: bool = False,
    device: str = "auto",
    seed: Optional[int] = None,
) -> None:
    if resume:
        trainer_factory = trainers.TrainerFactory(checkpoint=path, device=device)

        print("[scripts.train.train_autoencoder] Resuming trainer config:")
        pprint(trainer_factory.config)

        trainer = trainer_factory()
    else:
        if seed is not None:
            random.seed(seed)

        if env_config is None:
            env_config = envs.load_config(policy_checkpoints[0])

        env_factory = envs.EnvFactory(config=env_config)
        encoder_factory = encoders.EncoderFactory(
            config=encoder_config, env=env_factory(), device=device
        )
        trainer_factory = trainers.TrainerFactory(
            path=path,
            config=trainer_config,
            encoder=encoder_factory(),
            policy_checkpoints=policy_checkpoints,
            device=device,
        )

        print("[scripts.train.train_autoencoder] Trainer config:")
        pprint(trainer_factory.config)
        print("\n[scripts.train.train_autoencoder] Encoder config:")
        pprint(encoder_factory.config)
        print("\n[scripts.train.train_agent] Env config:")
        pprint(env_factory.config)
        print("\n[scripts.train.train_autoencoder] Policy checkpoints:")
        pprint(policy_checkpoints)
        print("")

        trainer = trainer_factory()

        trainer.path.mkdir(parents=True, exist_ok=overwrite)
        configs.save_git_hash(trainer.path)
        trainer_factory.save_config(trainer.path)
        encoder_factory.save_config(trainer.path)
        env_factory.save_config(trainer.path)

    trainer.train()


def main(args: argparse.Namespace) -> None:
    collect(**vars(args))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-configs", "-e", nargs="+", help="Paths to env configs")
    parser.add_argument("--path", "-p", required=True)
    parser.add_argument("--overwrite", action="store_true", default=False)
    parser.add_argument("--seed", type=int, help="Random seed")

    main(parser.parse_args())
