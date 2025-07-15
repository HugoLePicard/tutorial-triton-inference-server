import os

from utils import *

path_source      = os.path.join(BASE_PATH, "data"  , "models")
path_destivation = os.path.join(BASE_PATH, "triton", "models")

folder_names = [name for name in os.listdir(path_source) if os.path.isdir(os.path.join(path_source, name))]

for name in folder_names:
    for version in ["1", "2"]:
        path = os.path.join(path_destivation, name, version)
        os.makedirs(path, exist_ok=True)