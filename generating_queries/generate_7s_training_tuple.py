import numpy as np

# import quaternion
import pandas as pd
import pickle
import os
from sklearn.neighbors import KDTree

from datasets.seven_scenes import TrainingTuple

import argparse


def gen_tuple(scene, root_dir):

    datasets_folder = "/home/david/datasets"
    root_name = root_dir
    root_dir = os.path.join(datasets_folder, root_dir)
    train_pose_dir = os.path.join(root_dir, "train", "pose")
    test_pose_dir = os.path.join(root_dir, "test", "pose")

    train_pose_list = os.listdir(train_pose_dir)
    test_pose_list = os.listdir(test_pose_dir)

    train_pose_list.sort()
    test_pose_list.sort()

    train_poses = []
    train_translations = []
    test_poses = []
    test_translations = []

    for i in range(len(train_pose_list)):
        R = np.loadtxt(os.path.join(train_pose_dir, train_pose_list[i]))
        train_poses.append(R)
        # rotations.append(quaternion.from_rotation_matrix(R[:3, :3]))
        train_translations.append(R[:3, -1])

    for i in range(len(test_pose_list)):
        R = np.loadtxt(os.path.join(test_pose_dir, test_pose_list[i]))
        test_poses.append(R)
        # rotations.append(quaternion.from_rotation_matrix(R[:3, :3]))
        test_translations.append(R[:3, -1])

    train_tree = KDTree(train_translations)
    test_tree = KDTree(test_translations)
    train_positive = train_tree.query_radius(train_translations, r=0.5)
    train_non_negative = train_tree.query_radius(train_translations, r=1.0)

    test_positive = train_tree.query_radius(test_translations, r=0.4)
    test_non_negative = train_tree.query_radius(test_translations, r=1.0)

    # ind_p = tree.query_radius(translations, r=0.15)
    # ind_non = tree.query_radius(translations, r=1.5)

    # ind_p = tree.query_radius(translations, r=0.25)
    # ind_non = tree.query_radius(translations, r=1.6)

    queries = {}
    for anchor_ndx in range(len(train_pose_list)):
        anchor_pos = train_poses[anchor_ndx]
        query = os.path.join(
            root_dir,
            "color",
            train_pose_list[anchor_ndx].replace("pose.txt", "color.png"),
        )
        # Extract timestamp from the filename
        scan_filename = os.path.split(query)[1]
        timestamp = int(os.path.splitext(scan_filename)[0][-12:-6])

        positives = []
        non_negatives = []
        positives = train_positive[anchor_ndx]
        non_negatives = train_non_negative[anchor_ndx]

        positives = positives[positives != anchor_ndx]
        # Sort ascending order
        positives = np.sort(positives)
        non_negatives = np.sort(non_negatives)

        print(len(positives))
        print(len(non_negatives))

        queries[anchor_ndx] = TrainingTuple(
            id=anchor_ndx,
            timestamp=timestamp,
            rel_scan_filepath=query,
            positives=positives,
            non_negatives=non_negatives,
            pose=anchor_pos,
        )
    file_path = os.path.join(root_dir, root_name + "_train" + "_dist.pickle")
    with open(file_path, "wb") as handle:
        pickle.dump(queries, handle, protocol=pickle.HIGHEST_PROTOCOL)

    queries = {}
    for anchor_ndx in range(len(test_pose_list)):
        anchor_pos = test_poses[anchor_ndx]
        query = os.path.join(
            root_dir,
            "color",
            test_pose_list[anchor_ndx].replace("pose.txt", "color.png"),
        )
        # Extract timestamp from the filename
        scan_filename = os.path.split(query)[1]
        timestamp = int(os.path.splitext(scan_filename)[0][-12:-6])

        positives = []
        non_negatives = []
        positives = test_positive[anchor_ndx]
        non_negatives = test_non_negative[anchor_ndx]

        positives = positives[positives != anchor_ndx]
        # Sort ascending order
        positives = np.sort(positives)
        non_negatives = np.sort(non_negatives)

        queries[anchor_ndx] = TrainingTuple(
            id=anchor_ndx,
            timestamp=timestamp,
            rel_scan_filepath=query,
            positives=positives,
            non_negatives=non_negatives,
            pose=anchor_pos,
        )
    file_path = os.path.join(root_dir, root_name + "_test" + "_dist.pickle")
    with open(file_path, "wb") as handle:
        pickle.dump(queries, handle, protocol=pickle.HIGHEST_PROTOCOL)


def main():
    parser = argparse.ArgumentParser(description="Generate Baseline training dataset")
    parser.add_argument(
        "--dataset_root", type=str, required=True, help="Dataset root folder"
    )
    args = parser.parse_args()
    print("Dataset root: {}".format(args.dataset_root))
    root_dir = args.dataset_root

    gen_tuple("train", root_dir)
    # gen_tuple("test", root_dir)


if __name__ == "__main__":
    main()
