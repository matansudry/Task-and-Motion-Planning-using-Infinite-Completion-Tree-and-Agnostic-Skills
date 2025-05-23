from typing import Dict, Set, List, Union
import symbolic

import itertools
from string import Template
from collections import defaultdict
from configs.base_config import PDDLConfig

from scripts.eval.task_gen import utils

MOVABLE_TYPES = {"box", "tool", "movable"}
UNMOVABLE_TYPES = {"receptacle", "unmovable"}

def num_inhand(state: Union[List[str], Set[str]]) -> int:
    """Count the number of objects in the gripper."""
    count = 0
    for predicate in state:
        if "inhand" in predicate:
            count += 1
    return count

def is_hook_on_rack(state: Union[List[str], Set[str]]) -> bool:
    """Return True if state has a hook on a rack."""
    for predicate in state:
        if predicate == "on(hook, rack)":
            return True
    return False

def generate_symbolic_states(
    object_types: Dict[str, str],
    rack_properties: Set[str] = {"aligned", "poslimit"},
    hook_on_rack: bool = True,
) -> List[List[str]]:
    """Generate all possible symbolic states over specified objects.

    Arguments:
        object_types: Dictionary of object names to their type.

    Returns:
        symbolic_states: List of valid symbolic states.
    """
    movable_objects = [
        obj for obj, obj_type in object_types.items() if obj_type in MOVABLE_TYPES
    ]
    unmovable_objects = [
        obj for obj, obj_type in object_types.items() if obj_type in UNMOVABLE_TYPES
    ]
    locations = ["nonexistent($movable)", "inhand($movable)"] + [
        f"on($movable, {obj})" for obj in unmovable_objects
    ]

    # Store possible locations of objects.
    object_locations: Dict[str, List[Set[str]]] = defaultdict(list)
    for obj in movable_objects:
        for loc in locations:
            object_locations[obj].append({Template(loc).substitute(movable=obj)})

    # Rack predicates.
    if "rack" in unmovable_objects:
        rack_predicates = {f"{p}(rack)" for p in rack_properties}
        rack_inworkspace = {"on(rack, table)", "inworkspace(rack)"}.union(
            rack_predicates
        )
        rack_beyondworkspace = {"on(rack, table)", "beyondworkspace(rack)"}.union(
            rack_predicates
        )
        object_locations["rack"].extend([rack_inworkspace, rack_beyondworkspace])

    symbolic_states: List[List[str]] = []
    for state in itertools.product(*object_locations.values()):
        state = set.union(*state)
        if num_inhand(state) > 1 or (not hook_on_rack and is_hook_on_rack(state)):
            continue

        # Filter out nonexistent predicates.
        state = [p for p in state if "nonexistent" not in p]
        symbolic_states.append(utils.sort_propositions(state))

    return symbolic_states

def parse_args(str_prop: str) -> List[str]:
    """Parses the arguments of a proposition string."""
    import re

    matches = re.match(r"[^\(]*\(([^\)]*)", str_prop)
    if matches is None:
        raise ValueError(f"Unable to parse objects from '{str_prop}'.")
    str_args = matches.group(1).replace(" ", "").split(",")
    return str_args

def generate_pddl_problem(
    problem_name: str,
    pddl_config: PDDLConfig,
    object_types: Dict[str, str],
    symbolic_state: List[str],
    save: bool = True,
):
    """Generate a PDDL problem file given a symbolic predicate state.

    Arguments:
        problem_name: PDDL problem name.
        pddl_cgf: PDDLConfig.
        object_types: Dictionary of object names to their type.
        symbolic_state: List of symbolic predicates
        save: Saves PDDL problem file to disk if set True.

    Returns:
        problem: PDDL problem.
    """
    problem: symbolic.Problem = symbolic.Problem("tmp", "workspace")
    for obj, obj_type in object_types.items():
        if obj == "table":
            continue
        problem.add_object(obj, obj_type)
    for prop in symbolic_state:
        problem.add_initial_prop(prop)

    pddl_problem_file = pddl_config.get_problem_file(problem_name)
    if save:
        with open(pddl_problem_file, "w") as f:
            f.write(str(problem))

    return problem

def get_states_to_primitives(
    states_to_actions: Dict[str, List[str]], primitive: str
) -> Dict[str, List[str]]:
    """Get mapping from states to specified primitive actions."""
    states_to_primitives: Dict[str, List[str]] = {}
    for state, actions in states_to_actions.items():
        states_to_primitives[state] = [a for a in actions if primitive in a]
    return states_to_primitives

def get_symbolic_actions(
    state: List[str],
    object_types: Dict[str, str],
    pddl_config: PDDLConfig,
) -> List[str]:
    """Compute symbolically valid actions in a given state."""
    problem_name = str(state)
    _ = generate_pddl_problem(
        problem_name=problem_name,
        pddl_config=pddl_config,
        object_types=object_types,
        symbolic_state=state,
        save=True,
    )
    pddl = symbolic.Pddl(
        pddl_config.pddl_domain_file,
        pddl_config.get_problem_file(problem_name),
    )
    actions = pddl.list_valid_actions(pddl.initial_state)
    return actions

def get_state_object_types(
    state: List[str], object_types: Dict[str, str]
) -> Dict[str, str]:
    """Return dictionary of objects to object types for objects in state."""
    state_objects = set()
    for prop in state:
        state_objects = state_objects.union(set(parse_args(prop)))
    return {
        obj: obj_type for obj, obj_type in object_types.items() if obj in state_objects
    }