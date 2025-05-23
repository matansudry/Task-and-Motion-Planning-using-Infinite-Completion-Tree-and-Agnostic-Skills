from typing import List, Optional, Type

import torch

from stap.networks.critics.base import Critic
from stap.networks.mlp import LFF, MLP, weight_init


def create_q_network(
    observation_space,
    action_space,
    hidden_layers,
    act,
    fourier_features: Optional[int],
    output_act: Optional[Type[torch.nn.Module]] = None,
) -> torch.nn.Module:
    if fourier_features is not None:
        lff = LFF(observation_space.shape[0] + action_space.shape[0], fourier_features)
        mlp = MLP(
            fourier_features,
            1,
            hidden_layers=hidden_layers,
            act=act,
            output_act=output_act,
        )
        return torch.nn.Sequential(lff, mlp)
    else:
        mlp = MLP(
            observation_space.shape[0] + action_space.shape[0],
            1,
            hidden_layers=hidden_layers,
            act=act,
            output_act=output_act,
        )
        return mlp


class ContinuousMLPCritic(Critic):
    def __init__(
        self,
        observation_space,
        action_space,
        hidden_layers=[256, 256],
        act=torch.nn.ReLU,
        num_q_fns=2,
        ortho_init=False,
        fourier_features: Optional[int] = None,
        output_act: Optional[Type[torch.nn.Module]] = None,
    ):
        super().__init__()

        self.qs = torch.nn.ModuleList(
            [
                create_q_network(
                    observation_space,
                    action_space,
                    hidden_layers,
                    act,
                    fourier_features,
                    output_act,
                )
                for _ in range(num_q_fns)
            ]
        )
        if ortho_init:
            self.apply(weight_init)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> List[torch.Tensor]:  # type: ignore
        """Predicts the expected value of the given (state, action) pair.

        Args:
            state: State.
            action: Action.

        Returns:
            Predicted expected value.
        """
        x = torch.cat((state, action), dim=-1)
        return [q(x).squeeze(-1) for q in self.qs]

    def predict(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Predicts the expected value of the given (state, action) pair.

        Args:
            state: State.
            action: Action.

        Returns:
            Predicted expected value.
        """
        qs = self.forward(state, action)
        return torch.min(torch.stack(qs), dim=0).values
