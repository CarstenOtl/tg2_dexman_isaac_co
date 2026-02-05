"""Package containing task implementations for various robotic environments."""

import os
import toml

from isaaclab_tasks.utils import import_packages
import gymnasium as gym

from . import agents
from .dextrah_fr3_agilehand_env import DextrahFR3AgilehandEnv
from .dextrah_fr3_agilehand_env_cfg import DextrahFR3AgilehandEnvCfg

##
# Register Gym environments.
##

gym.register(
    id="dextrah_fr3_agilehand",
    entry_point="dextrah_lab.tasks.fr3_agilehand.dextrah_fr3_agilehand_env:DextrahFR3AgilehandEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": DextrahFR3AgilehandEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_lstm_cfg.yaml",
    },
)

# The blacklist is used to prevent importing configs from sub-packages
#_BLACKLIST_PKGS = ["utils"]
# Import all configs in this package
#import_packages(__name__, _BLACKLIST_PKGS)
