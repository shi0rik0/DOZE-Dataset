import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
from astar import astar_search
from util import my_round
import json
import warnings
import pickle

ASSET_PATH = config.ASSET_PATH
EPISODE_TYPE = config.EPISODE_TYPE
SCENE_ID = config.SCENE_ID
OUTPUT_DIR = config.OUTPUT_DIR

with open(os.path.join(OUTPUT_DIR, "visited.pkl"), "rb") as f:
    visited = pickle.load(f)

x_min = min([x for x, z in visited])
x_max = max([x for x, z in visited])
z_min = min([z for x, z in visited])
z_max = max([z for x, z in visited])

# 创建一个大小为 (x_max - x_min + 1, z_max - z_min + 1) 的网格
grid = [[1 for _ in range(z_max - z_min + 1)] for _ in range(x_max - x_min + 1)]
# 将 visited 中的坐标转换为网格坐标
for x, z in visited:
    grid[x - x_min][z - z_min] = 0

episode_path = os.path.join(
    ASSET_PATH, "episodes", EPISODE_TYPE, f"DOZE_{SCENE_ID}.json"
)
with open(episode_path, "r") as f:
    episodes = json.load(f)

paths = {}
for episode in episodes:
    id_ = episode["id"]
    initial_position = episode["initial_position"]
    initial_position = (
        my_round(initial_position["x"]) - x_min,
        my_round(initial_position["z"]) - z_min,
    )
    goal_position = episode["shortest_path"][-1]
    goal_position = (
        my_round(goal_position["x"]) - x_min,
        my_round(goal_position["z"]) - z_min,
    )
    path = astar_search(grid, initial_position, goal_position)
    if path:
        # 将路径坐标转换为实际坐标
        path = [(x + x_min, z + z_min) for x, z in path]
        initial_position = (
            initial_position[0] + x_min,
            initial_position[1] + z_min,
        )
        paths[id_] = dict(
            id=id_,
            path=path,
            episode_data=dict(
                initial_position=initial_position,
                initial_rotation=dict(
                    x=0,
                    y=episode["initial_orientation"],
                    z=0,
                ),
                goal_object=episode["goal_object"],
            ),
        )
    else:
        warnings.warn(f"未找到路径: {initial_position} -> {goal_position}")

with open(os.path.join(OUTPUT_DIR, "paths.pkl"), "wb") as f:
    pickle.dump(paths, f)
