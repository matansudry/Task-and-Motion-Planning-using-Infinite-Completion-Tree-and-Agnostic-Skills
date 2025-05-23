from planning.low_level.diffusion_low_level_planner import DiffusionLowLevelPlanner
from planning.low_level.stap_low_level_planner import STAPLowLevelPlanner
from planning.low_level.gsc_low_level_planner import GSCLowLevelPlanner

Low_LEVEL_PLANNERS = {
    "diffusion": DiffusionLowLevelPlanner,
    "stap": STAPLowLevelPlanner,
    "gsc": GSCLowLevelPlanner,
}