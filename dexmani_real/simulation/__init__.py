from dexmani_real.simulation.constructor import add_base_components, setup_scene
from dexmani_real.simulation.sim_adapter import SimRobotConfig, SimRobotInterface
from dexmani_real.simulation.xarm7_xhand import XArm7XHand, XArm7_XHand  # XArm7_XHand is deprecated

__all__ = [
    "SimRobotInterface",
    "SimRobotConfig",
    "XArm7XHand",
    "XArm7_XHand",  # deprecated alias
    "setup_scene",
    "add_base_components",
]
