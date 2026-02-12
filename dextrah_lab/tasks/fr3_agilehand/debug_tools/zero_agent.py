"""Zero-action agent for the FR3 Tekken ADoF (agilehand) environment.

Boots Isaac Sim, creates the ``dextrah_fr3_agilehand`` gym env, and steps it
with zero actions so you can visually verify the scene, check observations,
rewards, and resets without needing a trained policy.

Usage
-----
# Headless (fast smoke-test):
python -m dextrah_lab.tasks.fr3_agilehand.zero_agent --headless --num_envs 2

# With GUI:
python -m dextrah_lab.tasks.fr3_agilehand.zero_agent --num_envs 4

# Override objects directory:
python -m dextrah_lab.tasks.fr3_agilehand.zero_agent --num_envs 2 --objects_dir test_object
"""

import argparse

from isaaclab.app import AppLauncher

# ── CLI ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="Zero-action agent for dextrah_fr3_agilehand environment."
)
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations.",
)
parser.add_argument(
    "--num_envs",
    type=int,
    default=2,
    help="Number of environments to simulate.",
)
parser.add_argument(
    "--objects_dir",
    type=str,
    default="test_object",
    help="Name of the objects directory under assets/ to load (e.g. test_object, visdex_objects).",
)
parser.add_argument(
    "--max_steps",
    type=int,
    default=0,
    help="Maximum env steps to run. 0 = unlimited (run until window is closed).",
)
parser.add_argument(
    "--print_every",
    type=int,
    default=60,
    help="Print observation / reward summary every N env steps.",
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch Kit / Isaac Sim before importing anything that needs USD / carb.
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── Imports that require a running Kit ───────────────────────────────────────
import gymnasium as gym
import torch

from isaaclab_tasks.utils import parse_env_cfg

# Register the fr3_agilehand gym id.
import dextrah_lab.tasks.fr3_agilehand.gym_setup  # noqa: F401


def main() -> None:
    """Run the zero-action agent loop."""
    # ── Build env config ─────────────────────────────────────────────────
    env_cfg = parse_env_cfg(
        "dextrah_fr3_agilehand",
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env_cfg.objects_dir = args_cli.objects_dir
    # Make sure the chosen objects_dir is accepted.
    if env_cfg.objects_dir not in env_cfg.valid_objects_dir:
        env_cfg.valid_objects_dir.append(env_cfg.objects_dir)

    # Disable ADR for a clean sanity check.
    env_cfg.enable_adr = False
    env_cfg.starting_adr_increments = 0

    # ── Create the environment ───────────────────────────────────────────
    env = gym.make("dextrah_fr3_agilehand", cfg=env_cfg)

    print("=" * 72)
    print("  Zero-Action Agent  —  dextrah_fr3_agilehand")
    print("=" * 72)
    print(f"  num_envs           : {env.unwrapped.num_envs}")
    print(f"  num_actions        : {env.unwrapped.num_actions}")
    print(f"  num_observations   : {env.unwrapped.num_observations}")
    print(f"  num_states (critic): {env.unwrapped.cfg.num_states}")
    print(f"  objects_dir        : {env_cfg.objects_dir}")
    print(f"  device             : {env.unwrapped.device}")
    print("=" * 72)

    # ── Reset ────────────────────────────────────────────────────────────
    obs_dict, info = env.reset()
    if isinstance(obs_dict, dict):
        obs = obs_dict.get("policy", obs_dict.get("obs", None))
    else:
        obs = obs_dict

    if obs is not None:
        print(f"\n[INFO] Initial obs shape : {obs.shape}")
        print(f"[INFO] Initial obs range : [{obs.min().item():.4f}, {obs.max().item():.4f}]")

    # Zero action tensor matching the action space.
    zero_actions = torch.zeros(
        env.unwrapped.num_envs,
        env.unwrapped.num_actions,
        device=env.unwrapped.device,
    )

    # ── Step loop ────────────────────────────────────────────────────────
    step = 0
    cumulative_reward = torch.zeros(env.unwrapped.num_envs, device=env.unwrapped.device)
    episode_lengths = torch.zeros(env.unwrapped.num_envs, device=env.unwrapped.device, dtype=torch.long)
    num_resets = 0
    print_every = max(1, args_cli.print_every)

    print("\n[INFO] Stepping with zero actions. Close the window (or Ctrl-C) to stop.\n")

    while simulation_app.is_running():
        with torch.inference_mode():
            obs_dict, reward, terminated, truncated, info = env.step(zero_actions)

        if isinstance(obs_dict, dict):
            obs = obs_dict.get("policy", obs_dict.get("obs", None))
        else:
            obs = obs_dict

        cumulative_reward += reward
        episode_lengths += 1
        step += 1

        # Count resets (done envs).
        dones = terminated | truncated
        n_done = dones.sum().item()
        if n_done > 0:
            num_resets += int(n_done)
            # Reset trackers for done envs.
            cumulative_reward[dones] = 0.0
            episode_lengths[dones] = 0

        # Periodic summary.
        if step % print_every == 0:
            obs_min = obs.min().item() if obs is not None else float("nan")
            obs_max = obs.max().item() if obs is not None else float("nan")
            print(
                f"[step {step:6d}]  "
                f"reward mean={reward.mean().item():+.4f}  "
                f"cum_reward mean={cumulative_reward.mean().item():+.4f}  "
                f"obs range=[{obs_min:.3f}, {obs_max:.3f}]  "
                f"dones={n_done}  "
                f"total_resets={num_resets}"
            )
            # Print success rate if available.
            if hasattr(env.unwrapped, "in_success_region"):
                sr = env.unwrapped.in_success_region.float().mean().item()
                print(f"           success_rate={sr:.4f}")

        # Optional step cap.
        if args_cli.max_steps > 0 and step >= args_cli.max_steps:
            print(f"\n[INFO] Reached --max_steps={args_cli.max_steps}. Stopping.")
            break

    # ── Cleanup ──────────────────────────────────────────────────────────
    env.close()
    print(f"\n[DONE] Ran {step} steps with {num_resets} total env resets.")


if __name__ == "__main__":
    main()
    simulation_app.close()
