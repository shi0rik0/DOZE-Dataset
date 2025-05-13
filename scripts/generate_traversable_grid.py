import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
from util import my_round
from ai2thor.controller import Controller
import pickle
import collections

GRID_SIZE = config.GRID_SIZE
ASSET_PATH = config.ASSET_PATH
OUTPUT_DIR = config.OUTPUT_DIR
SCENE_TYPE = config.SCENE_TYPE
SCENE_ID = config.SCENE_ID

controller_config = dict(
    local_executable_path=os.path.join(
        ASSET_PATH, "scenes", SCENE_TYPE, f"DOZE_{SCENE_TYPE}_{SCENE_ID}.x86_64"
    ),
    visibilityDistance=1.5,
    # step sizes
    gridSize=GRID_SIZE,
    snapToGrid=True,
    rotateStepDegrees=90,
    # image modalities
    renderDepthImage=False,
    renderInstanceSegmentation=False,
    # camera properties
    width=1920,
    height=1080,
    fieldOfView=90,
)

controller = Controller(agentMode="default", **controller_config)


def get_position():
    controller.step("Done")
    p = controller.last_event.metadata["agent"]["position"]
    return tuple(map(my_round, [p["x"], p["z"]]))


def move(direction: int, distance: int = 1):
    if not 0 <= direction <= 3:
        raise ValueError("Direction must be between 0 and 3")
    controller.step("Teleport", rotation=dict(x=0, y=direction * 90, z=0))
    controller.step("MoveAhead", moveMagnitude=distance * GRID_SIZE)


def move_until_end(direction: int):
    if not 0 <= direction <= 3:
        raise ValueError("Direction must be between 0 and 3")
    last_pos = get_position()
    step_size = 1
    controller.step("Teleport", rotation=dict(x=0, y=direction * 90, z=0))
    while True:
        controller.step("MoveAhead", moveMagnitude=step_size * GRID_SIZE)
        new_pos = get_position()
        if new_pos == last_pos:
            break
        last_pos = new_pos
        step_size *= 2
    while True:
        controller.step("MoveAhead", moveMagnitude=step_size * GRID_SIZE)
        new_pos = get_position()
        if new_pos == last_pos:
            if step_size <= 1:
                break
            step_size //= 2
        last_pos = new_pos


def teleport(x: int, z: int):
    controller.step("Done")
    y = controller.last_event.metadata["agent"]["position"]["y"]
    controller.step("Teleport", position=dict(x=x * GRID_SIZE, y=y, z=z * GRID_SIZE))


visited = set()
queue = collections.deque()  # 使用双端队列作为 BFS 的队列


def bfs():
    # 1. 获取初始位置
    start_x, start_z = get_position()

    # 2. 如果初始位置未被访问过，则将其加入队列和 visited 集合
    if (start_x, start_z) not in visited:
        visited.add((start_x, start_z))
        queue.append((start_x, start_z))
        print(f"Starting BFS at: {start_x}, {start_z}", flush=True)  # 初始节点信息

    # 3. 当队列不为空时，循环处理
    while queue:
        # 3a. 从队列头部取出一个节点 (当前节点)
        curr_x, curr_z = queue.popleft()
        print(
            f"Visiting: {curr_x}, {curr_z}", flush=True
        )  # 访问当前节点 (与 DFS 中的 print 对应)

        # 3b. 探索当前节点的所有邻居 (通过四个方向的移动)
        for direction in range(4):  # 假设有4个方向
            # 重要: 在尝试移动到下一个邻居之前，
            # 确保我们从当前处理的节点 (curr_x, curr_z) 出发。
            # 这对应了 DFS 版本中在循环内部的 teleport 调用。
            teleport(curr_x, curr_z)

            move(direction)  # 执行移动
            next_x, next_z = get_position()

            if next_x == curr_x:
                if next_z > curr_z:
                    new_pos = [(curr_x, i) for i in range(curr_z + 1, next_z + 1)]
                else:
                    new_pos = [(curr_x, i) for i in range(next_z, curr_z)]
            elif next_z == curr_z:
                if next_x > curr_x:
                    new_pos = [(i, curr_z) for i in range(curr_x + 1, next_x + 1)]
                else:
                    new_pos = [(i, curr_z) for i in range(next_x, curr_x)]

            # 3c. 如果新位置未被访问过，则标记为已访问并加入队列
            for i, j in new_pos:
                if (i, j) not in visited:
                    visited.add((i, j))
                    queue.append((i, j))
                # print(f"  Added to queue: ({next_x}, {next_z}) from ({curr_x}, {curr_z})") # 可选的调试信息


# 假设 get_position() 在开始时能提供一个有效的起始点
# 例如，如果需要手动设置初始位置：
# current_x, current_z = 0, 0 # 假设的初始位置

# 启动 BFS
bfs()

os.makedirs(OUTPUT_DIR, exist_ok=True)
with open(os.path.join(OUTPUT_DIR, "visited.pkl"), "wb") as f:
    pickle.dump(visited, f)
print("Visited coordinates saved to visited.pkl")
