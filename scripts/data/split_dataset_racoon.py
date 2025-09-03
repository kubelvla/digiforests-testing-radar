# MIT License
#
# Copyright (c) 2025 Meher Malladi, Luca Lobefaro, Tiziano Guadagnino, Cyrill Stachniss.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import typer
from pathlib import Path
import os
import re
import yaml
import math


from digiforests_dataloader.utils.io import write_json
from digiforests_dataloader.utils.logging import logger

app = typer.Typer(rich_markup_mode="markdown")

# Regex for filenames: cloud_[stamp]_to_[stamp]_at_[x]_[y]_[z]_.pcd
FILENAME_RE = re.compile(
r"cloud_(?P<t1>\d+\.\d+)_to_(?P<t2>\d+\.\d+)_at_(?P<x>-?\d+\.\d+)_(-?\d+\.\d+)_(-?\d+\.\d+)_\.pcd"
)

def parse_filename(fname):
    m = FILENAME_RE.match(fname)
    if not m:
        return None
    x, y, z = float(m.group(3)), float(m.group(4)), float(m.group(5))
    return x, y, z

def load_cuboids(yaml_path, sample_index):
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)


    cuboids = []
    sample = data.get("dataset",[]).get("samples",[])[sample_index]
    cuboids_yaml = sample.get("labels", []).get("ground-truth", []).get("attributes", []).get("annotations", [])

    for cub in cuboids_yaml:
        # Segments.ai cuboid format example:
        # { "position": {"x":..,"y":..,"z":..}, "dimensions": {"x":..,"y":..,"z":..}, "yaw":.., "category":.. }
        pos = cub["position"]
        dims = cub["dimensions"]
        yaw = cub.get("yaw", 0.0)
        category = cub.get("category_id", 1)
        cuboids.append({
                        "center": (pos["x"], pos["y"], pos["z"]),
                        "dims": (dims["x"], dims["y"], dims["z"]),
                        "yaw": yaw,
                        "category": category
        })
    return cuboids

def point_in_cuboid(point, cuboid):
    px, py, pz = point
    cx, cy, cz = cuboid["center"]
    dx, dy, dz = cuboid["dims"]
    yaw = cuboid["yaw"]


    # Translate point to cuboid frame
    tx, ty = px - cx, py - cy


    # Rotate point by -yaw around z-axis
    cos_y, sin_y = math.cos(-yaw), math.sin(-yaw)
    rx = cos_y * tx - sin_y * ty
    ry = sin_y * tx + cos_y * ty
    rz = pz - cz


    # Check within half-dimensions
    return (
        abs(rx) <= dx / 2 and
        abs(ry) <= dy / 2 and
        abs(rz) <= dz / 2
    )

def split_files(pcd_dir, yaml_path, sample_index):
    cuboids = load_cuboids(yaml_path, sample_index)

    train, val, test = [], [], []

    for file in list(pcd_dir.glob("*.pcd")):
        parsed = parse_filename(file.name)
        if not parsed:
            continue


        assigned = False
        for cub in cuboids:
            if point_in_cuboid(parsed, cub):
                if cub["category"] == 2:
                    val.append(file)
                elif cub["category"] == 3:
                    test.append(file)
                assigned = True
                break


        if not assigned:
            train.append(file)

    return train, val, test

@app.command()
def split(
    raw_folder: Path = typer.Argument(
        ..., help="Path to the raw data folder containing DigiForests dataset."
    ),
    cuboids_json_file: Path = typer.Argument(
        ..., help="Path to the cuboids json file to split."
    ),
    output_fp: Path | None = typer.Option(
        None, help="Optional path to save the data split JSON file."
    ),
):
    """
    Split the DigiForests dataset into train, validation, test, and prediction sets.

    This function organizes the DigiForests dataset into predefined splits based on
    experiment folders. It generates a JSON file containing file paths and statistics
    for each split.

    \n\n**Args:**\n
    - `raw_folder`: Root directory of the DigiForests dataset containing experiment folders.\n
    - `output_fp`: Custom path to save the output JSON file. If None, saves to raw_folder/data_split.json.

    \n\n**Splits:**\n
    - Train: Primary training data from multiple seasons.\n
    - Validation: Held-out data for model tuning.\n
    - Test: Unseen data for final model evaluation.\n
    - Prediction: Specific subset (Spring 2023) for inference tasks.

    \n\n**Output JSON Structure:**\n
    - File counts for each split\n
    - Train/Val ratios\n
    - File paths for each split\n
    - Combined trainval set

    \n\n**Note:**\n
    - Split ratios are calculated based on train and validation sets only.\n
    - The function assumes a specific folder structure within the raw_folder.
    """

    exp_folders_and_sample_index = [
    #    ("2024-05/all_clouds_skip4",1),
    #    ("2024-06/all_clouds_skip4",0),
    #    ("2024_05/single_scan_skip7",1)
    #     ("2024_05/five_scans_skip_2", 1),
    #     ("2024_06/five_scans_skip_2", 0)
        ("2024_05/five_scans_skip_2_trunk_inflated", 1),
        ("2024_06/five_scans_skip_2_trunk_inflated", 0)
    ]

    train_files_abs = []
    val_files_abs = []
    test_files_abs = []

    for exp_folder, sample_index in exp_folders_and_sample_index:
        train, val, test = split_files((raw_folder / exp_folder), cuboids_json_file, sample_index)
        train_files_abs.extend(train)
        val_files_abs.extend(val)
        test_files_abs.extend(test)

    train_files = [file_path.relative_to(raw_folder) for file_path in train_files_abs]
    val_files = [file_path.relative_to(raw_folder) for file_path in val_files_abs]
    test_files = [file_path.relative_to(raw_folder) for file_path in test_files_abs]
    pred_files = test_files[::10]

    exp_files = [train_files, val_files, test_files, pred_files]

    trainval_files = []

    num_train, num_val, num_test, num_pred = map(len, exp_files)

    trainval_files.extend(train_files)
    trainval_files.extend(val_files)
    num_trainval = len(trainval_files)

    data_dict = {
        "num_train": num_train,
        "num_val": num_val,
        "num_test": num_test,
        "num_pred": num_pred,
        "train_ratio": len(train_files) / num_trainval,
        "val_ratio": len(val_files) / num_trainval,
        "train": train_files,
        "val": val_files,
        "test": test_files,
        "pred": pred_files,
        "trainval": trainval_files,
    }
    json_fp = output_fp or raw_folder / "data_split.json"
    write_json(data_dict, json_fp, sort_keys=False, overwrite=True)

    logger.info(
        f"train ratio: {data_dict['train_ratio']}, val_ratio: {data_dict['val_ratio']}, trainval files: {num_trainval}"
    )
    logger.info(
        "num_train",
        num_train,
        "num_val",
        num_val,
        "num_test",
        num_test,
        "num_pred",
        num_pred,
    )
    logger.info(f"{json_fp} written with the data split")


if __name__ == "__main__":
    app()
