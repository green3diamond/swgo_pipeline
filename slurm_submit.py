import os
import subprocess
import argparse
import textwrap
from datetime import datetime
from typing import Dict, Optional, Sequence, List

DEFAULT_SLURM_CONFIG: Dict[str, object] = {
    "job-name": "swgo_training",
    "partition": None,            # Let Slurm default unless overridden
    "nodes": 1,
    "ntasks-per-node": 1,
    # Prefer using --gres for typed GPUs; keep this optional for sites that want it
    "gpus-per-task": None,        # Only emit if explicitly set
    "mem": "32G",
    "time": "02:00:00",
    "gres": "gpu:1",                 # e.g., "gpu:nvidia_a100-pcie-40gb:1"
}

def _here_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__))

def _resolve_training_script_path(training_script_path: Optional[str]) -> str:
    """Resolve absolute path to swgo_trainv3.py.
    If not provided, assume it lives alongside this file.
    """
    if training_script_path:
        return os.path.abspath(os.path.expanduser(training_script_path))
    guess = os.path.join(_here_dir(), "swgo_trainv3.py")
    return os.path.abspath(guess)

def _get_flag_value(argv: Sequence[str], name: str) -> Optional[str]:
    """Return value for --name from argv if present; supports --name=value and --name value."""
    flag = f"--{name}"
    for i, tok in enumerate(argv):
        if tok.startswith(flag + "="):
            return tok.split("=", 1)[1]
        if tok == flag and i + 1 < len(argv):
            return argv[i + 1]
    return None

def _sanitize_path(p: Optional[str]) -> Optional[str]:
    if p is None:
        return None
    return os.path.abspath(os.path.expanduser(p))

def _make_sbatch_script(training_script_path: str,
                        passthrough_args: str,
                        slurm_config: Dict[str, object],
                        log_dir: str) -> str:
    job_name = slurm_config["job-name"]
    output_log = os.path.join(log_dir, f"{job_name}.out")
    error_log = os.path.join(log_dir, f"{job_name}.err")

    lines: List[str] = [
        "#!/bin/bash",
        "#",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --output={output_log}",
        f"#SBATCH --error={error_log}",
        "#",
    ]

    if slurm_config.get("partition"):
        lines.append(f"#SBATCH --partition={slurm_config['partition']}")

    lines += [
        f"#SBATCH --nodes={slurm_config['nodes']}",
        f"#SBATCH --ntasks-per-node={slurm_config['ntasks-per-node']}",
        f"#SBATCH --mem={slurm_config['mem']}",
        f"#SBATCH --time={slurm_config['time']}",
    ]

    # Optional GPU requests
    if slurm_config.get("gres"):
        lines.append(f"#SBATCH --gres={slurm_config['gres']}")
    if slurm_config.get("gpus-per-task"):
        lines.append(f"#SBATCH --gpus-per-task={slurm_config['gpus-per-task']}")

    # Body
    lines += [
        "",
        'echo "========================================================="',
        'echo "Starting job on $(hostname)"',
        'echo "Job ID: $SLURM_JOB_ID"',
        'echo "Timestamp: $(date)"',
        'echo "========================================================="',
        'echo ""',
        "",
        "# --- Environment Setup (IMPORTANT: MODIFY FOR YOUR SYSTEM) ---",
        "module load python",
        "mamba activate /n/holylabs/arguelles_delgado_lab/Lab/tamboOpt_env",
        "export COMET_API_KEY='ycVI3E7BxdmtYD7d4NvJlGINB'",
        "export COMET_WORKSPACE='hamzahanif2210'",
        "export COMET_PROJECT_NAME='swgo_train_New'",
        "",
        f'CMD="python {training_script_path} {passthrough_args}"',
        'echo "Executing command:"',
        'echo "$CMD"',
        'echo ""',
        "$CMD",
        "rc=$?",
        'echo ""',
        'echo "========================================================="',
        'echo "Job finished with exit code $rc"',
        'echo "Timestamp: $(date)"',
        'echo "========================================================="',
        "exit $rc",
    ]

    return textwrap.dedent("\n".join(lines))

def submit_to_slurm(passthrough_argv: Sequence[str],
                    training_script_path: Optional[str] = None,
                    slurm_config: Optional[Dict[str, object]] = None,
                    log_dir: str = "slurm_logs") -> int:
    """Generate an sbatch script for swgo_trainv3.py and submit it.

    Args:
        passthrough_argv: The argument list to pass through to the training script.
        training_script_path: Path to swgo_trainv3.py. If None, auto-resolve next to this file.
        slurm_config: Optional overrides for DEFAULT_SLURM_CONFIG.
        log_dir: Directory to write Slurm stdout/err logs. May be overridden by --main_dir/--run_name passthrough.
    Returns:
        Exit code (0 if 'sbatch' submission succeeded).
    """
    cfg = dict(DEFAULT_SLURM_CONFIG)
    if slurm_config:
        cfg.update(slurm_config)

    # Auto-resolve training script
    training_script_path = _resolve_training_script_path(training_script_path)

    if not os.path.exists(training_script_path):
        print(f"Error: Training script '{training_script_path}' not found.")
        return 2

    # If the training script was passed --main_dir and --run_name, use that for log_dir
    main_dir_cli = _get_flag_value(passthrough_argv, "main_dir")
    run_name_cli = _get_flag_value(passthrough_argv, "run_name")
    if main_dir_cli and run_name_cli:
        main_dir_abs = _sanitize_path(main_dir_cli)
        log_dir = os.path.join(main_dir_abs, run_name_cli, "slurm_logs")  # keep grouped with the run
    os.makedirs(log_dir, exist_ok=True)

    passthrough_args = " ".join(passthrough_argv)

    sbatch_script = _make_sbatch_script(training_script_path, passthrough_args, cfg, log_dir)

    # Compute and print the log file paths (same logic as in _make_sbatch_script)
    job_name = cfg["job-name"]
    output_log = os.path.join(log_dir, f"{job_name}.out")
    error_log = os.path.join(log_dir, f"{job_name}.err")

    # Save SBATCH script
    sbatch_file = os.path.join(log_dir, f"{job_name}.sbatch")
    with open(sbatch_file, "w") as f:
        f.write(sbatch_script)


    print("Logs will be written to:")
    print(f"  stdout: {output_log}")
    print(f"  stderr: {error_log}")
    print("-----------------------------")

    try:
        process = subprocess.run(
            ["sbatch"],
            input=sbatch_script.encode("utf-8"),
            capture_output=True,
            check=True,
        )
        print("--- Submission Result ---")
        print(process.stdout.decode())
        print("Job submitted successfully!")
        return 0
    except FileNotFoundError:
        print("\nError: 'sbatch' command not found. Are you on a Slurm login node?")
        return 127
    except subprocess.CalledProcessError as e:
        print("\nError submitting job to Slurm.")
        print(e.stderr.decode())
        return e.returncode or 1

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Submit a swgo_trainv3.py job to a Slurm cluster.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent('''            Examples:
          python slurm_submit.py -- --run_name my_first_run --epochs 20
          python slurm_submit.py --partition gpu --gres gpu:nvidia_a100-pcie-40gb:1 -- --run_name a100_try1 --epochs 20

        Note: Use the "--" separator so all following args go to swgo_trainv3.py.
        '''),
    )
    # Quick overrides for common Slurm fields
    parser.add_argument("--job-name", dest="job_name", default=None)
    parser.add_argument("--partition", dest="partition", default=None)
    parser.add_argument("--time", dest="time", default=None)
    parser.add_argument("--mem", dest="mem", default=None)
    parser.add_argument("--gres", dest="gres", default=None,
                        help="Slurm --gres string, e.g. gpu:nvidia_a100-pcie-40gb:1")
    parser.add_argument("--gpus-per-task", dest="gpus_per_task", default=None,
                        help="Slurm --gpus-per-task value (avoid if using typed --gres)")
    parser.add_argument("--training-script", dest="training_script", default=None,
                        help="Path to swgo_trainv3.py (defaults to file next to slurm_submit.py)")

    # Everything after '--' goes directly to swgo_trainv3.py
    parser.add_argument("trainer_args", nargs=argparse.REMAINDER,
                        help="Arguments passed to swgo_trainv3.py (prefix with -- after the separator)")

    args = parser.parse_args()

    overrides: Dict[str, object] = {}
    if args.job_name:
        overrides["job-name"] = args.job_name
    if args.partition:
        overrides["partition"] = args.partition
    if args.time:
        overrides["time"] = args.time
    if args.mem:
        overrides["mem"] = args.mem
    if args.gres:
        overrides["gres"] = args.gres
    if args.gpus_per_task:
        overrides["gpus-per-task"] = args.gpus_per_task

    trainer_args = args.trainer_args or []
    if trainer_args and trainer_args[0] == "--":
        trainer_args = trainer_args[1:]

    return submit_to_slurm(
        trainer_args,
        training_script_path=args.training_script,
        slurm_config=overrides or None,
    )

if __name__ == "__main__":
    raise SystemExit(main())