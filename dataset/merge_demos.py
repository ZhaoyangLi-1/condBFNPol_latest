import os
import re
import random
from pathlib import Path
import shutil

PATTERN = re.compile(r"^traj_?\d+$")


def find_all_traj_dirs(root_dir: str | Path):
    root = Path(root_dir)
    traj_dirs = []

    for p in root.rglob("*"):
        if p.is_dir() and PATTERN.match(p.name):
            traj_dirs.append(p)

    traj_dirs.sort(key=lambda x: int(re.findall(r"\d+", x.name)[0]))
    return traj_dirs


def reindex_and_copy(src_root, dst_root, sample_size=50):
    src_root = Path(src_root)
    dst_root = Path(dst_root)

    traj_dirs = find_all_traj_dirs(src_root)
    total = len(traj_dirs)

    print(f"Found {total} traj folders")

    if total < sample_size:
        raise ValueError(f"Only {total} traj folders found, cannot sample {sample_size}")

    sampled_trajs = random.sample(traj_dirs, sample_size)

    sampled_trajs.sort(key=lambda x: int(re.findall(r"\d+", x.name)[0]))

    dst_root.mkdir(parents=True, exist_ok=True)

    for new_idx, old_path in enumerate(sampled_trajs):
        new_name = f"traj{new_idx}"
        new_path = dst_root / new_name

        print(f"Copying {old_path} -> {new_path}")
        shutil.copytree(old_path, new_path)

    print("Done.")


random.seed(42)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

root_path = os.path.join(CURRENT_DIR, "BFN_data_raw")
new_path = os.path.join(CURRENT_DIR, "BFN_data")

if os.path.exists(new_path):
    shutil.rmtree(new_path)

reindex_and_copy(root_path, new_path, sample_size=50)
