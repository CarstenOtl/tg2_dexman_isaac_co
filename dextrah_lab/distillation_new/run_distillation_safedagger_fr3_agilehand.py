"""Script to perform SafeDagger student-teacher distillation for FR3 Agilehand"""

import argparse
import sys

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="SafeDagger distillation for FR3 Agilehand.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--teacher", type=str, default=None, help="Teacher checkpoint to use")
parser.add_argument("--network", type=str, default=None, help="Student network checkpoint to resume from")
parser.add_argument("--play_policy", type=bool, default=False, help="Play a distilled policy.")
parser.add_argument("--data_aug", action="store_true", default=False, help="Whether to use data augmentation for student")
parser.add_argument("--mono", action="store_true", default=False, help="Use monocular instead of stereo (default: stereo)")
parser.add_argument("--no_transformer", action="store_true", default=False, help="Disable transformer student (default: transformer)")
parser.add_argument("--vanilla_dagger", action="store_true", default=False, help="Use vanilla DAgger (no unsafe override) instead of SafeDAgger. Both use weighted L2 loss.")
parser.add_argument("--bc", action="store_true", default=False, help="Pure behavior cloning: teacher always steps the env, student only learns the mapping.")
parser.add_argument("--unsafe_l2_threshold", type=float, default=2.0, help="L2 threshold for SafeDagger teacher intervention (default: 2.0)")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True


# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


"""Rest everything follows."""

import gymnasium as gym
import os
from datetime import datetime
import pathlib

from rl_games.algos_torch import model_builder

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

from distillation_safedagger import SafeDagger
import dextrah_lab.tasks.fr3_agilehand.gym_setup

from dextrah_lab.distillation.a2c_with_aux_depth import A2CBuilder as A2CWithAuxDepthBuilder
from dextrah_lab.distillation.a2c_with_aux_cnn import A2CBuilder as A2CWithAuxCNNBuilder
from dextrah_lab.distillation.a2c_with_aux_cnn_stereo import A2CBuilder as A2CWithAuxCNNStereoBuilder
from dextrah_lab.distillation.a2c_with_aux_cnn_stereo_recon import A2CBuilder as A2CWithAuxCNNStereoReconBuilder
from dextrah_lab.distillation.a2c_with_pretrain import A2CBuilder as A2CWithPretrainBuilder
from dextrah_lab.distillation.a2c_stereo_transformer import A2CBuilder as A2CStereoTransformerBuilder
from dextrah_lab.distillation.a2c_mono_resnet import A2CBuilder as A2CMonoResnetBuilder
from dextrah_lab.distillation.a2c_mono_transformer import A2CBuilder as A2CMonoTransformerBuilder


@hydra_task_config(args_cli.task, "rl_games_cfg_entry_point")
def main(env_cfg, agent_cfg: dict):
    """ Performs SafeDagger distillation for FR3 Agilehand. """
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    ov_env = env.env

    parent_path = str(pathlib.Path(__file__).parent.parent.parent.resolve())
    agent_cfg_folder = "dextrah_lab/tasks/fr3_agilehand/agents"

    use_stereo = not args_cli.mono
    use_transformer = not args_cli.no_transformer
    vision_tag = "stereo" if use_stereo else "mono"
    arch_tag = "transformer" if use_transformer else "cnn"

    if args_cli.bc and args_cli.vanilla_dagger:
        raise ValueError("Cannot use both --behavior-cloning and --vanilla_dagger.")

    if use_stereo and use_transformer:
        student_yaml = "rl_games_ppo_stereo_transformer.yaml"
    elif not use_stereo and use_transformer:
        student_yaml = "rl_games_ppo_mono_transformer.yaml"
    elif use_stereo and not use_transformer:
        student_yaml = "rl_games_ppo_lstm_scratch_cnn_aux_stereo.yaml"
    else:
        student_yaml = "rl_games_ppo_lstm_scratch_cnn_aux.yaml"

    student_cfg = os.path.join(parent_path, agent_cfg_folder, student_yaml)

    teacher_cfg = os.path.join(
        parent_path,
        agent_cfg_folder,
        "rl_games_ppo_lstm_cfg.yaml"
    )

    # Student checkpoint from --network flag
    student_ckpt = None
    if args_cli.network is not None:
        student_ckpt = args_cli.network
        if not os.path.isabs(student_ckpt):
            student_ckpt = os.path.join(parent_path, student_ckpt)

    # Determine teacher checkpoint path
    teacher_ckpt = None
    if not args_cli.play_policy:
        if args_cli.teacher is not None:
            teacher_ckpt = args_cli.teacher
            if not os.path.isabs(teacher_ckpt):
                teacher_ckpt = os.path.join(parent_path, "pretrained_ckpts", teacher_ckpt)
        else:
            teacher_ckpt = os.path.join(parent_path, "pretrained_ckpts/fr3_agilehand_teacher.pth")

    train_dir = "runs"
    if args_cli.bc:
        method_tag = "bc"
    elif args_cli.vanilla_dagger:
        method_tag = "dagger"
    else:
        method_tag = "safedagger"
    experiment_name = (
        f"dextrah-fr3-agilehand-{method_tag}-{vision_tag}-{arch_tag}"
        + datetime.now().strftime("_%d-%H-%M-%S")
    )
    experiment_dir = os.path.join(train_dir, experiment_name)
    nn_dir = os.path.join(experiment_dir, "nn")
    summaries_dir = os.path.join(experiment_dir, "summaries")

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(nn_dir, exist_ok=True)
    os.makedirs(summaries_dir, exist_ok=True)

    dagger_config = {
        "student": {
            "cfg": student_cfg,
            "ckpt": student_ckpt,
            "obs_type": "policy",
            "data_aug": args_cli.data_aug,
        },
        "teacher": {
            "cfg": teacher_cfg,
            "ckpt": teacher_ckpt,
            "obs_type": "expert_policy",
        },
        "imitation_loss_type": "l2",  # weighted L2 for both SafeDagger and DAgger (DextrAH-RGB paper)
        "play_policy": args_cli.play_policy,
        "disable_unsafe_override": args_cli.vanilla_dagger or args_cli.bc,
        "behavior_cloning": args_cli.bc,
        "unsafe_l2_threshold": args_cli.unsafe_l2_threshold,
    }

    model_builder.register_network("a2c_aux_depth_enc", A2CWithAuxDepthBuilder)
    model_builder.register_network("a2c_aux_cnn_net", A2CWithAuxCNNBuilder)
    model_builder.register_network("a2c_aux_cnn_net_stereo", A2CWithAuxCNNStereoBuilder)
    model_builder.register_network("a2c_aux_cnn_net_stereo_recon", A2CWithAuxCNNStereoReconBuilder)
    model_builder.register_network("a2c_aux_pretrain", A2CWithPretrainBuilder)
    model_builder.register_network("a2c_stereo_transformer", A2CStereoTransformerBuilder)
    model_builder.register_network("a2c_mono_resnet", A2CMonoResnetBuilder)
    model_builder.register_network("a2c_mono_transformer", A2CMonoTransformerBuilder)

    dagger = SafeDagger(env, dagger_config, summaries_dir=summaries_dir, nn_dir=nn_dir, max_iterations=args_cli.max_iterations)
    dagger.distill()
    dagger.save(f"dextrah_student_{method_tag}_{vision_tag}_{arch_tag}")


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()