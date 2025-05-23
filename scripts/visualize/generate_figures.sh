#!/bin/bash

function run_cmd {
    echo ""
    echo "${CMD}"
    ${CMD}
}

function visualize_results {
    args=""
    args="${args} --path ${PLANNER_OUTPUT_ROOT}"
    args="${args} --envs ${TASKS[@]}"
    args="${args} --methods ${METHODS[@]}"
    if [ ! -z "${FIGURE_NAME}" ]; then
        args="${args} --name ${FIGURE_NAME}"
    fi
    CMD="python scripts/visualize/generate_planning_figure.py ${args}"
    run_cmd
}

# Setup.
input_path="plots"

# Pybullet experiments.
exp_name="planning"
PLANNER_OUTPUT_ROOT="${input_path}/${exp_name}"

METHODS=(
    "irl_policy_cem"
    "policy_cem"
    "train0/daf_random_cem_light"
    "train1/daf_random_cem_light"
    "train2/daf_random_cem_light"
    "random_cem"
    "policy_shooting"
    "random_shooting"
    "greedy"
)

TASKS=(
# Domain 1: Hook Reach
    "hook_reach/task0"
    "hook_reach/task1"
    "hook_reach/task2"
# Domain 2: Constrained Packing
    "constrained_packing/task0"
    "constrained_packing/task1"
    "constrained_packing/task2"
# Domain 3: Rearrangement Push
    "rearrangement_push/task0"
    "rearrangement_push/task1"
    "rearrangement_push/task2"
)

FIGURE_NAME="planning-result"
visualize_results