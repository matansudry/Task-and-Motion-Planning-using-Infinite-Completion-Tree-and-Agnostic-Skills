import pathlib
from typing import Any, Dict, List, Optional, Union

from stap import envs
from stap.utils import configs


class EnvFactory(configs.Factory[envs.Env]):
    """Env factory."""

    def __init__(
        self,
        config: Union[str, pathlib.Path, Dict[str, Any]],
    ):
        """Creates the env factory from an env config or policy checkpoint.

        Args:
            config: Env config path or dict.
        """
        super().__init__(config, "env", envs)

        if issubclass(self.cls, envs.VariantEnv):
            self._variants = [
                EnvFactory(env_config) for env_config in self.kwargs["variants"]
            ]

    def __call__(self, *args, multiprocess: bool = False, **kwargs) -> envs.Env:
        """Creates an env instance.

        Args:
            *args: Env constructor args.
            multiprocess: Whether to wrap the env in a ProcessEnv.
            **kwargs: Env constructor kwargs.

        Returns:
            Env instance.
        """
        if multiprocess:
            raise NotImplementedError

        if issubclass(self.cls, envs.VariantEnv):
            variants = [env_factory(*args, **kwargs) for env_factory in self._variants]
            return super().__call__(variants=variants)

        return super().__call__(*args, **kwargs)


def load(
    config: Optional[Union[str, pathlib.Path, Dict[str, Any]]] = None,
    checkpoint: Optional[Union[str, pathlib.Path]] = None,
    multiprocess: bool = False,
    **kwargs,
) -> envs.Env:
    """Loads the agent from an env config or policy checkpoint.

    Args:
        config: Optional env config path or dict. Must be set if checkpoint is
            None.
        checkpoint: Optional policy checkpoint path.
        multiprocess: Whether to run the env in a separate process.
        kwargs: Additional env constructor kwargs.

    Returns:
        Env instance.
    """
    if config is None:
        if checkpoint is None:
            raise ValueError("Env config or checkpoint must be specified")
        config = load_config(checkpoint)

    env_factory = EnvFactory(config)
    return env_factory(multiprocess=multiprocess, **kwargs)


def load_config(path: Union[str, pathlib.Path]) -> Dict[str, Any]:
    """Loads an env config from path.

    Args:
        path: Path to the config, config directory, or checkpoint.

    Returns:
        Env config dict.
    """
    return configs.load_config(path, "env")
