import os
import sys
import time
import subprocess
from typing import List, Tuple

# -------------------------------------------------------------
# Configuration
# -------------------------------------------------------------
# Image tags: must match file names in images/ (without extension)
TAGS: List[str] = [
    "bark",
    "beehive",
    "coffee",
    "rose",
    "stucco",
    "water",
]

# Model depths to train
LAYERS: List[int] = [1, 2, 3]

# Optionally restrict which physical GPUs to use.
# Example: VISIBLE_GPUS = ["0", "1", "2", "3"]
# By default (None), the script will read CUDA_VISIBLE_DEVICES
# from the environment (if any) or otherwise use all GPUs.
VISIBLE_GPUS = None  # type: ignore


def get_physical_gpus() -> List[str]:
    """Return a list of physical GPU ids (as strings) to use.

    Priority:
      1) VISIBLE_GPUS variable if not None;
      2) CUDA_VISIBLE_DEVICES env var if set;
      3) torch.cuda.device_count() if CUDA is available;
      4) [] if no GPU is available.
    """
    if VISIBLE_GPUS is not None:
        return [str(g) for g in VISIBLE_GPUS]

    env_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if env_visible:
        return [g.strip() for g in env_visible.split(",") if g.strip() != ""]

    try:
        import torch

        if torch.cuda.is_available():
            return [str(i) for i in range(torch.cuda.device_count())]
    except Exception:
        # torch may not be installed, fall back to CPU
        pass

    return []


def build_experiment_list() -> List[Tuple[int, str]]:
    """Build the list of (layer, tag) experiments to run."""
    experiments: List[Tuple[int, str]] = []
    for tag in TAGS:
        for layer in LAYERS:
            experiments.append((layer, tag))
    return experiments


def run_sequential(experiments: List[Tuple[int, str]]) -> None:
    """Run all experiments sequentially on CPU (or default device)."""
    print("No GPU detected. Running experiments sequentially on CPU / default device.")
    for i, (layer, tag) in enumerate(experiments, 1):
        print(f"[CPU] ({i}/{len(experiments)}) Running tag='{tag}', layer={layer} ...", flush=True)
        cmd = [sys.executable, "deep_frame.py", "--layer", str(layer), "--tag", tag]
        ret = subprocess.call(cmd)
        if ret != 0:
            print(f"[CPU] Experiment tag='{tag}', layer={layer} exited with code {ret}.")


def run_parallel_on_gpus(experiments: List[Tuple[int, str]], physical_gpu_ids: List[str]) -> None:
    """Schedule experiments across given GPUs, running up to len(physical_gpu_ids)
    experiments in parallel.
    """
    num_gpus = len(physical_gpu_ids)
    assert num_gpus > 0

    print(f"Detected {num_gpus} GPU(s): {physical_gpu_ids}")
    print(f"Total experiments: {len(experiments)}")

    # Logical GPU indices 0..num_gpus-1 correspond to physical IDs in physical_gpu_ids
    free_gpu_slots = list(range(num_gpus))  # logical indices
    running = []  # list of dicts with process info

    exp_idx = 0
    total_exps = len(experiments)

    try:
        while exp_idx < total_exps or running:
            # Launch new jobs while we have free GPUs and remaining experiments
            while free_gpu_slots and exp_idx < total_exps:
                logical_gpu = free_gpu_slots.pop(0)
                physical_gpu = physical_gpu_ids[logical_gpu]
                layer, tag = experiments[exp_idx]
                exp_idx += 1

                env = os.environ.copy()
                # Each child process only sees a single GPU, mapped to cuda:0 inside deep_frame.py
                env["CUDA_VISIBLE_DEVICES"] = str(physical_gpu)

                cmd = [sys.executable, "deep_frame.py", "--layer", str(layer), "--tag", tag]
                print(
                    f"[LAUNCH] tag='{tag}', layer={layer} on physical GPU {physical_gpu} "
                    f"({exp_idx}/{total_exps})",
                    flush=True,
                )

                proc = subprocess.Popen(cmd, env=env)
                running.append(
                    {
                        "proc": proc,
                        "logical_gpu": logical_gpu,
                        "physical_gpu": physical_gpu,
                        "layer": layer,
                        "tag": tag,
                    }
                )

            if not running:
                continue

            # Sleep a bit, then check which jobs finished
            time.sleep(10)
            still_running = []
            for job in running:
                ret = job["proc"].poll()
                if ret is None:
                    still_running.append(job)
                else:
                    print(
                        f"[FINISH] tag='{job['tag']}', layer={job['layer']} "
                        f"on GPU {job['physical_gpu']} with exit code {ret}",
                        flush=True,
                    )
                    free_gpu_slots.append(job["logical_gpu"])
            free_gpu_slots.sort()
            running = still_running

    except KeyboardInterrupt:
        print("\nKeyboardInterrupt received, terminating all running experiments ...")
        for job in running:
            try:
                job["proc"].terminate()
            except Exception:
                pass
        print("All child processes have been signaled to terminate.")


def main() -> None:
    # Ensure images directory exists and contains the tags we expect
    if not os.path.isdir("images"):
        print("Error: 'images/' directory not found in current working directory.")
        sys.exit(1)

    image_files = {os.path.splitext(f)[0] for f in os.listdir("images")}
    missing_tags = [t for t in TAGS if t not in image_files]
    if missing_tags:
        print("Warning: the following tags do not have corresponding files in images/: ")
        print("   ", ", ".join(missing_tags))

    experiments = build_experiment_list()

    physical_gpus = get_physical_gpus()
    if not physical_gpus:
        run_sequential(experiments)
    else:
        run_parallel_on_gpus(experiments, physical_gpus)


if __name__ == "__main__":
    main()
