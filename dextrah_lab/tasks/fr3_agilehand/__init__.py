## this is not used.... task registreation is in gym_setup.py

# """Package containing task implementations for FR3 Tekken ADoF."""

# import gymnasium as gym

# from . import agents
# from .dextrah_tg2_inspirehand_env import DextrahFR3AgilehandEnv
# from .dextrah_tg2_inspirehand_env_cfg import DextrahFR3AgilehandEnvCfg

# ##
# # Register Gym environments.
# ##

# gym.register(
#     id="dextrah_fr3_agilehand",
#     entry_point="dextrah_lab.tasks.fr3_agilehand.dextrah_tg2_inspirehand_env:DextrahFR3AgilehandEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": DextrahFR3AgilehandEnvCfg,
#         "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_lstm_cfg.yaml",
#     },
# )
