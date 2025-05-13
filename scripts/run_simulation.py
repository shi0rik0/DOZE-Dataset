import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
from util import my_round
from ai2thor.controller import Controller
import cv2
import json
import uuid
import pickle

SCENE_TYPE = config.SCENE_TYPE
SCENE_ID = config.SCENE_ID
ASSET_PATH = config.ASSET_PATH
GRID_SIZE = config.GRID_SIZE
OUTPUT_DIR = config.OUTPUT_DIR
OUTPUT_RESOLUTION = config.OUTPUT_RESOLUTION

with open(os.path.join(OUTPUT_DIR, "paths.pkl"), "rb") as f:
    paths = pickle.load(f)


def rotate_left(controller, degrees=90, steps=10, step_delay=0.05):
    step_size = degrees / steps
    for _ in range(steps):
        controller.step("RotateLeft", degrees=step_size)
        # time.sleep(step_delay)


def rotate_right(controller, degrees=90, steps=10, step_delay=0.05):
    step_size = degrees / steps
    for _ in range(steps):
        controller.step("RotateRight", degrees=step_size)
        # time.sleep(step_delay)


def move_ahead(controller, step_delay=0.05):
    controller.step("MoveAhead")
    # time.sleep(step_delay)


def write_img(bgr_img, dir) -> str:
    """将图像写入指定目录，生成一个随机UUID作为文件名，然后返回文件名"""
    file_name = str(uuid.uuid4()) + ".jpg"
    file_path = os.path.join(dir, file_name)
    cv2.imwrite(file_path, bgr_img)
    return file_name


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
    width=OUTPUT_RESOLUTION[0],
    height=OUTPUT_RESOLUTION[1],
    fieldOfView=90,
)

controller = Controller(agentMode="default", **controller_config)

agent_pos = controller.last_event.metadata["agent"]["position"]
print(agent_pos)
print({**agent_pos, "y": 0.5})
# 增加新的摄像头
controller.step(
    "AddThirdPartyCamera",
    position={
        **agent_pos,
        "y": 0.4,  # 将高度设置为0.4
        "z": agent_pos["z"] - 0.05,  # 稍微将摄像头往前移一点，不然会拍摄到机体
    },
    rotation=dict(x=0, y=180, z=0),
    fieldOfView=90,
)


img1 = controller.last_event.cv2img
img2 = controller.last_event.third_party_camera_frames[0]
img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)
# show the image
# cv2.imshow("img1", img1)
# cv2.imshow("img2", img2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

controller.step("MoveAhead", moveMagnitude=2)

agent_pos = controller.last_event.metadata["agent"]["position"]
print(agent_pos)
print({**agent_pos, "y": 0.5})
# 增加新的摄像头
controller.step(
    "UpdateThirdPartyCamera",
    position={**agent_pos, "y": 0.4},
    rotation=dict(x=0, y=180, z=0),
    fieldOfView=90,
)

img3 = controller.last_event.cv2img
img4 = controller.last_event.third_party_camera_frames[0]
img4 = cv2.cvtColor(img4, cv2.COLOR_RGB2BGR)
# show the image
cv2.imshow("img1", img2)
cv2.imshow("img2", img4)
cv2.waitKey(0)
cv2.destroyAllWindows()

raise Exception("stop here")


def get_position():
    controller.step("Done")
    p = controller.last_event.metadata["agent"]["position"]
    return tuple(map(my_round, [p["x"], p["z"]]))


def move_to(x, z):
    rotation = {
        (0, 1): 0,
        (1, 0): 90,
        (0, -1): 180,
        (-1, 0): 270,
    }
    pos = get_position()
    rot = round(controller.last_event.metadata["agent"]["rotation"]["y"])
    diff = (x - pos[0], z - pos[1])
    target_rot = rotation[diff]
    while rot != target_rot:
        rotate_left(controller)
        rot = round(controller.last_event.metadata["agent"]["rotation"]["y"])
    move_ahead(controller)


infos = {}

image_dir = os.path.join(OUTPUT_DIR, "images")
os.makedirs(image_dir, exist_ok=True)


def execute(path_dict):
    path = path_dict["path"]
    id_ = path_dict["id"]
    info = dict(steps=[])
    x = path[0][0] * GRID_SIZE
    z = path[0][1] * GRID_SIZE
    y = controller.last_event.metadata["agent"]["position"]["y"]
    rot = controller.last_event.metadata["agent"]["rotation"]
    info["steps"].append(
        dict(
            position=[x, y, z],
            rotation=[rot["x"], rot["y"], rot["z"]],
            frame=write_img(controller.last_event.cv2img, image_dir),
        )
    )
    controller.step("Teleport", position=dict(x=x, y=y, z=z))
    for p in path[1:]:
        move_to(*p)
        x = p[0] * GRID_SIZE
        z = p[1] * GRID_SIZE
        rot = controller.last_event.metadata["agent"]["rotation"]
        info["steps"].append(
            dict(
                position=[x, y, z],
                rotation=[rot["x"], rot["y"], rot["z"]],
                frame=write_img(controller.last_event.cv2img, image_dir),
            )
        )
    info["episode_id"] = id_
    for k, v in path_dict["episode_data"].items():
        info[k] = v
    info["initial_position"] = dict(
        x=info["initial_position"][0] * GRID_SIZE,
        y=y,
        z=info["initial_position"][1] * GRID_SIZE,
    )
    infos[id_] = info


for path in paths.values():
    execute(path)
    print(f"Executed path: {path}")

with open(os.path.join(OUTPUT_DIR, "infos.json"), "w") as f:
    json.dump(infos, f, indent=4)
