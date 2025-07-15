import os
import shutil

from utils import *

path_source      = os.path.join(BASE_PATH, "data", "models")
path_destination = os.path.join(BASE_PATH, "triton", "models")

folder_names = [name for name in os.listdir(path_source) if os.path.isdir(os.path.join(path_source, name))]

for name in folder_names:
    path_source_model      = os.path.join(path_source, name, "model.onnx")
    path_destination_dir   = os.path.join(path_destination, name, "1")
    path_destination_model = os.path.join(path_destination_dir, "model.onnx")

    os.makedirs(path_destination_dir, exist_ok=True)

    if os.path.isfile(path_source_model):
        shutil.copy2(path_source_model, path_destination_model)
        print(f"Copied: {path_source_model} → {path_destination_model}")
    else:
        print(f"Skipped (not found): {path_source_model}")