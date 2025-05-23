import pathlib
from typing import (
    Any,
    Dict,
    Literal,
    Optional,
    Iterable,
    List,
    Sequence,
    Tuple,
    Union,
    Callable,
)

import numpy as np
import torch
import yaml

from stap import agents, dynamics, envs, networks, planners
from stap.dynamics import Dynamics, LatentDynamics, load as load_dynamics
from stap.utils import configs, spaces, tensors, timing, recording
from stap.envs.pybullet.table.primitives import Null


class PlannerFactory(configs.Factory):
    """Planner factory."""

    def __init__(
        self,
        config: Union[str, pathlib.Path, Dict[str, Any]],
        env: envs.Env,
        policy_checkpoints: Optional[
            Sequence[Optional[Union[str, pathlib.Path]]]
        ] = None,
        policies: Optional[Sequence[agents.Agent]] = None,
        scod_checkpoints: Optional[Sequence[Optional[Union[str, pathlib.Path]]]] = None,
        dynamics_checkpoint: Optional[Union[str, pathlib.Path]] = None,
        dynamics: Optional[Dynamics] = None,
        device: str = "auto",
    ):
        """Creates the planner factory from a planner_config.

        Args:
            config: Planner config path or dict.
            env: Sequential env.
            policy_checkpoints: Policy checkpoint paths if required.
            policies: Optional policies to replace policy_checkpoints.
            scod_checkpoints: SCOD checkpoint paths if required.
            dynamics_checkpoint: Dynamics checkpoint path if required.
            dynamics: Optional dynamics to replace dynamics_checkpoints.
            device: Torch device.
        """

        def replace_config(config, old: str, new: str):
            config_yaml: str = yaml.dump(config)
            config_yaml = config_yaml.replace(old, new)
            config = yaml.safe_load(config_yaml)
            return config

        super().__init__(config, "planner", planners)

        if scod_checkpoints is None:
            if policy_checkpoints is not None:
                scod_checkpoints = [None] * len(policy_checkpoints)
            else:
                scod_checkpoints = [None] * len(self.config["agent_configs"])
        if policy_checkpoints is None:
            policy_checkpoints = [None] * len(self.config["agent_configs"])
        else:
            assert len(scod_checkpoints) == len(
                policy_checkpoints
            ), "All policies must have SCOD checkpoints"
            for idx_policy, (policy_checkpoint, scod_checkpoint) in enumerate(
                zip(policy_checkpoints, scod_checkpoints)
            ):
                # Get policy config from checkpoint
                if policy_checkpoint is None:
                    continue
                agent_config = str(
                    pathlib.Path(policy_checkpoint).parent / "agent_config.yaml"
                )
                self.config["agent_configs"][idx_policy] = replace_config(
                    self.config["agent_configs"][idx_policy],
                    "{AGENT_CONFIG}",
                    agent_config,
                )
                # Optionally get scod config from checkpoint
                if scod_checkpoint is None:
                    continue
                scod_config = str(
                    pathlib.Path(scod_checkpoint).parent / "scod_config.yaml"
                )
                self.config["agent_configs"][idx_policy] = replace_config(
                    self.config["agent_configs"][idx_policy],
                    "{SCOD_CONFIG}",
                    scod_config,
                )

        # Get dynamics config from checkpoint.
        if dynamics_checkpoint is not None:
            dynamics_config = str(
                pathlib.Path(dynamics_checkpoint).parent / "dynamics_config.yaml"
            )
            self.config["dynamics_config"] = replace_config(
                self.config["dynamics_config"], "{DYNAMICS_CONFIG}", dynamics_config
            )

        maybe_policies = (
            [None] * len(self.config["agent_configs"]) if policies is None else policies
        )
        policies = [
            agents.load(
                config=agent_config,
                env=env,
                checkpoint=ckpt,
                scod_checkpoint=scod_ckpt,
                policy=policy,
            )
            for agent_config, ckpt, scod_ckpt, policy in zip(
                self.config["agent_configs"],
                policy_checkpoints,
                scod_checkpoints,
                maybe_policies,
            )
        ]

        if dynamics is None:
            # Make sure all policy checkpoints are not None for dynamics.
            dynamics_policy_checkpoints: Optional[List[Union[str, pathlib.Path]]] = []
            for policy_checkpoint in policy_checkpoints:
                if policy_checkpoint is None:
                    dynamics_policy_checkpoints = None
                    break
                assert dynamics_policy_checkpoints is not None
                dynamics_policy_checkpoints.append(policy_checkpoint)

            dynamics = load_dynamics(
                config=self.config["dynamics_config"],
                checkpoint=dynamics_checkpoint,
                policies=policies,
                policy_checkpoints=dynamics_policy_checkpoints,
                env=env,
                device=device,
            )

        self.kwargs["policies"] = policies
        self.kwargs["dynamics"] = dynamics
        if isinstance(dynamics, LatentDynamics):
            dynamics.plan_mode()
        self.kwargs["device"] = device


def load(
    config: Union[str, pathlib.Path, Dict[str, Any]],
    env: envs.Env,
    policies: Optional[Sequence[agents.Agent]] = None,
    policy_checkpoints: Optional[Sequence[Optional[Union[str, pathlib.Path]]]] = None,
    scod_checkpoints: Optional[Sequence[Optional[Union[str, pathlib.Path]]]] = None,
    dynamics_checkpoint: Optional[Union[str, pathlib.Path]] = None,
    dynamics: Optional[Dynamics] = None,
    device: str = "auto",
    **kwargs,
) -> planners.Planner:
    """Loads the planner from config.

    Args:
        config: Planner config path or dict.
        env: Sequential env.
        policy_checkpoints: Policy checkpoint paths if required.
        policies: Optional policies to replace policy_checkpoints.
        scod_checkpoints: SCOD checkpoint paths if required.
        dynamics_checkpoint: Dynamics checkpoint path if required.
        dynamics: Optional dynamics to replace dynamics_checkpoints.
        device: Torch device.
        **kwargs: Planner constructor kwargs.

    Returns:
        Planner instance.
    """
    planner_factory = PlannerFactory(
        config=config,
        env=env,
        policy_checkpoints=policy_checkpoints,
        scod_checkpoints=scod_checkpoints,
        dynamics_checkpoint=dynamics_checkpoint,
        device=device,
    )
    return planner_factory(**kwargs)


# TODO: states.ndim isn't necessarily 2.
@tensors.batch(dims=2)
def evaluate_trajectory(
    value_fns: Iterable[
        Union[networks.critics.Critic, networks.critics.ProbabilisticCritic]
    ],
    decode_fns: Iterable[Callable[[torch.Tensor], torch.Tensor]],
    states: torch.Tensor,
    actions: Optional[torch.Tensor] = None,
    q_value: bool = True,
    clip_success: bool = True,
    unc_metric: Optional[str] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Evaluates probability of success for the given trajectory.

    Args:
        value_fns: List of T value functions.
        decoders: List of T decoders.
        states: [batch_dims, T + 1, state_dims] trajectory states.
        actions: [batch_dims, T, state_dims] trajectory actions.
        q_value: Whether to use state-action values (True) or state values (False).
        clip_success: Whether to clip successes between [0, 1].
        unc_metric: Uncertainty metric if value_fn outputs a distribution.

    Returns:
        (Trajectory success probabilities [batch_size],
         values [batch_size, T], value uncertainty metric [batch_size, T]) 2-tuple.
    """
    # Compute step success probabilities.
    p_successes = torch.zeros(
        (states.shape[0], states.shape[1] - 1),
        dtype=torch.float32,
        device=states.device,
    )

    p_successes_unc = torch.zeros_like(p_successes)
    if q_value:
        assert actions is not None
        for t, (value_fn, decode_fn) in enumerate(zip(value_fns, decode_fns)):
            policy_state = decode_fn(states[:, t])
            dim_action = int(torch.sum(~torch.isnan(actions[0, t])).cpu().item())
            action = actions[:, t, :dim_action]
            if isinstance(value_fn, networks.critics.Critic):
                p_successes[:, t] = value_fn.predict(policy_state, action)

            # Value functions that output a torch distribution.
            elif isinstance(value_fn, networks.critics.ProbabilisticCritic):
                if unc_metric is None:
                    raise ValueError(
                        "Must specify unc_metric if value_fn outputs a distribution."
                    )
                p_distribution = value_fn.forward(policy_state, action)
                p_successes[:, t] = p_distribution.mean
                p_successes_unc[:, t] = getattr(p_distribution, unc_metric)

            # Ensemble OOD detector critics with a detect property.
            if isinstance(value_fn, networks.critics.EnsembleDetectorCritic):
                p_successes_unc[:, t] = value_fn.detect
    else:
        raise NotImplementedError

    if clip_success:
        p_successes = torch.clip(p_successes, min=0, max=1)

    # Combine probabilities.
    p_success = torch.exp(torch.log(p_successes).sum(dim=-1))
    p_successes = torch.exp(torch.log(p_successes[:,0]))

    return p_success, p_successes, p_successes_unc


def evaluate_plan(
    env: envs.Env,
    action_skeleton: Sequence[envs.Primitive],
    actions: np.ndarray,
    gif_path: Optional[Union[str, pathlib.Path]] = None,
) -> np.ndarray:
    """Evaluates the given open-loop plan.

    Args:
        env: Sequential env.
        action_skeleton: List of primitives.
        actions: Planned actions [T, A].
        gif_path: Optional path to save a rendered gif.

    Returns:
        Rewards received at each timestep.
    """
    if gif_path is not None:
        env.record_start()

    # Iterate over plan.
    rewards = np.zeros(len(action_skeleton), dtype=np.float32)
    for t, primitive in enumerate(action_skeleton):
        # Execute action.
        env.set_primitive(primitive)
        action = actions[t, : env.action_space.shape[0]]
        _, reward, _, _, _ = env.step(action)
        rewards[t] = reward

        if reward == 0.0:
            break

    if gif_path is not None:
        env.record_stop()
        gif_path = pathlib.Path(gif_path)
        if (rewards == 0.0).any():
            gif_path = gif_path.parent / f"{gif_path.name}_fail{gif_path.suffix}"
        env.record_save(gif_path, reset=True)

    return rewards


def get_printable_object_relationships_str(
    obj_rels: List[str], max_row_length: int = 60
) -> None:
    """
    Get printable object relationships string.
    """
    overall_str: str = ""
    curr_line: str = "obj_rel: "
    # add curr_line to env._recording_text before getting too long
    # then add "\n" to the front of new curr_line
    for obj_rel in obj_rels:
        if len(curr_line) + len(obj_rel) + 1 > max_row_length:
            overall_str += curr_line + "\n"
            curr_line = ""
        curr_line += obj_rel + ", "

    overall_str += curr_line
    return overall_str


def vizualize_predicted_plan(
    save_path_suffix: Union[int, str],
    env: envs.Env,
    action_skeleton: Sequence[envs.Primitive],
    plan: planners.PlanningResult,
    path: pathlib.Path,
    custom_recording_text: Optional[str] = None,
    object_relationships_list: Optional[List[List[str]]] = None,
    file_extensions: Optional[List[Literal["gif", "mp4"]]] = None,
) -> None:
    """Visualize the predicted trajectory of a task and motion plan."""
    import pybullet as p

    assert isinstance(
        env, envs.pybullet.TableEnv
    ), "vizualize_predicted_plan only supports pybullet.TableEnv"
    state_id: int = p.saveState()
    recorder = recording.Recorder()
    recorder.start()

    for i, (primitive, predicted_state, action) in enumerate(
        zip(action_skeleton, plan.states[:-1], plan.actions)
    ):
        env.set_primitive(primitive)
        if custom_recording_text is not None:
            if isinstance(custom_recording_text, list):
                env._recording_text = custom_recording_text[i]
            else:
                env._recording_text = custom_recording_text
        else:
            env._recording_text = (
                "Action: ["
                + ", ".join([f"{a:.2f}" for a in primitive.scale_action(action)])
                + "]"
            )

        if object_relationships_list is not None:
            env._recording_text += "\n" + get_printable_object_relationships_str(
                object_relationships_list[i]
            )

        env.set_observation(predicted_state)
        recorder.add_frame(frame=env.render())

    env._recording_text = ""
    # add final frame and text
    env.set_primitive(Null())
    if custom_recording_text is not None:
        if isinstance(custom_recording_text, list):
            env._recording_text = custom_recording_text[-1]
        else:
            env._recording_text = custom_recording_text
    if object_relationships_list is not None:
        env._recording_text += "\n" + get_printable_object_relationships_str(
            object_relationships_list[-1]
        )
    env.set_observation(plan.states[-1])
    recorder.add_frame(frame=env.render())

    recorder.stop()
    for i, file_extension in enumerate(file_extensions):
        recorder.save(
            path / f"predicted_trajectory_{save_path_suffix}.{file_extension}",
            reset=i == len(file_extensions) - 1,
        )
    # prevent visualization from corrupting the env state
    p.restoreState(state_id)


def run_open_loop_planning(
    env: envs.Env,
    action_skeleton: Sequence[envs.Primitive],
    planner: planners.Planner,
    timer: Optional[timing.Timer] = None,
    gif_path: Optional[Union[str, pathlib.Path]] = None,
    record_timelapse: bool = False,
) -> Tuple[np.ndarray, planners.PlanningResult, Optional[List[float]]]:
    if isinstance(planner.dynamics, dynamics.OracleDynamics):
        state = env.get_state()

    if record_timelapse and gif_path is not None:
        env.record_start("timelapse", mode="timelapse")

    # Plan.
    if timer is not None:
        timer.tic("planner")
    plan = planner.plan(env.get_observation(), env.action_skeleton)
    t_planner = None if timer is None else timer.toc("planner")

    if record_timelapse and gif_path is not None:
        env.record_stop("timelapse", mode="timelapse")
        env.record_save(gif_path, reset=True)

    if isinstance(planner.dynamics, dynamics.OracleDynamics):
        env.set_state(state)

    # Execute plan.
    rewards = evaluate_plan(env, action_skeleton, plan.actions, gif_path=gif_path)

    if isinstance(planner.dynamics, dynamics.OracleDynamics):
        env.set_state(state)

    return rewards, plan, None if t_planner is None else [t_planner]


def run_closed_loop_planning(
    env: envs.Env,
    action_skeleton: Sequence[envs.Primitive],
    planner: planners.Planner,
    timer: Optional[timing.Timer] = None,
    gif_path: Optional[Union[str, pathlib.Path]] = None,
    record_timelapse: bool = False,
) -> Tuple[np.ndarray, planners.PlanningResult, Optional[List[float]]]:
    """Runs closed-loop planning.

    Args:
        env: Sequential env.
        action_skeleton: List of primitives.
        actions: Planned actions [T, A].
        gif_path: Optional path to save a rendered gif.

    Returns:
        Rewards received at each timestep.
    """
    if isinstance(planner.dynamics, dynamics.OracleDynamics):
        raise ValueError(
            "Do not run closed-loop planning with OracleDynamics! Open-loop gets the same results."
        )

    if gif_path is not None:
        env.record_start()

    T = len(action_skeleton)
    rewards = np.zeros(T, dtype=np.float32)
    actions = spaces.null(planner.dynamics.action_space, batch_shape=T)
    states = spaces.null(planner.dynamics.state_space, batch_shape=T + 1)
    values = np.full(T, float("nan"), dtype=np.float32)
    visited_actions = spaces.null(planner.dynamics.action_space, batch_shape=(T, T))
    visited_states = spaces.null(planner.dynamics.state_space, batch_shape=(T, T + 1))
    p_visited_success = np.full(T, float("nan"), dtype=np.float32)
    visited_values = np.full((T, T), float("nan"), dtype=np.float32)

    observation = env.get_observation()
    t_planner: Optional[List[float]] = None if timer is None else []
    for t, primitive in enumerate(action_skeleton):
        env.set_primitive(primitive)

        # Plan.
        if timer is not None:
            timer.tic("planner")
        plan = planner.plan(observation, action_skeleton[t:])
        if t_planner is not None and timer is not None:
            t_planner.append(timer.toc("planner"))

        # Execute first action.
        observation, reward, _, _, _ = env.step(
            plan.actions[0, : env.action_space.shape[0]]
        )

        rewards[t] = reward
        visited_actions[t, t:] = plan.actions
        visited_states[t, t:] = plan.states
        p_visited_success[t] = plan.p_success
        visited_values[t, t:] = plan.values

        if reward == 0.0:
            actions[t:] = plan.actions
            states[t:] = plan.states
            values[t:] = plan.values
            break

        actions[t] = plan.actions[0]
        states[t : t + 1] = plan.states[:1]
        values[t] = plan.values[0]
        # ensure last state has something in it
        states[t + 1 : t + 2] = observation

    p_success = np.exp(np.log(values).sum())

    if gif_path is not None:
        env.record_stop()
        gif_path = pathlib.Path(gif_path)
        if (rewards == 0.0).any():
            gif_path = gif_path.parent / f"{gif_path.name}_fail{gif_path.suffix}"
        env.record_save(gif_path, reset=True)

    plan = planners.PlanningResult(
        actions=actions,
        states=states,
        p_success=p_success,
        values=values,
        visited_actions=visited_actions,
        visited_states=visited_states,
        p_visited_success=p_visited_success,
        visited_values=visited_values,
    )

    return rewards, plan, t_planner
