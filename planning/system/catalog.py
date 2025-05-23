from planning.system.system_class import SystemPlanner
from planning.system.els_system import ELSSystemPlanner
from planning.system.els_v2_system import ELSv2SystemPlanner
from planning.system.our_tamp import TAMPPlanner
from planning.system.els_v3_system import ELSv3SystemPlanner

SYSTEM_CATALOG = {
    "els": ELSSystemPlanner,
    "system": SystemPlanner,
    "els_v2": ELSv2SystemPlanner,
    "tamp": TAMPPlanner,
    "els_v3": ELSv3SystemPlanner,
}
