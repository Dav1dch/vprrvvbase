# PointNetVLAD datasets: based on Oxford RobotCar and Inhouse
# Code adapted from PointNetVLAD repo: https://github.com/mikacuy/pointnetvlad

import os
import pickle
from sklearn.neighbors import KDTree


def output_to_file(output, base_path, filename):
    file_path = os.path.join(base_path, filename)
    with open(file_path, "wb") as handle:
        pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("Done ", filename)




# train_pickle = pickle.load(
#     open("/home/david/datasets/fire/pickle/fire_train_dist.pickle", "rb")
# )
# test_pickle = pickle.load(
#     open("/home/david/datasets/fire/pickle/fire_test_dist.pickle", "rb")
# )


import numpy as np


def construct_query_and_database_sets(root_name):
    root_dir = os.path.join("/home/david/datasets/", root_name)
    rgb_dir = os.path.join(root_dir, "color")


    database_pose_dir = os.path.join(root_dir, "train/pose")
    query_pose_dir = os.path.join(root_dir , "test/pose")
    query_pose_list = os.listdir(query_pose_dir)
    query_pose_list.sort()

    database_pose_list = os.listdir(database_pose_dir)
    database_pose_list.sort()
    query_pose = []
    database_pose = []
    for i in range(len(database_pose_list)):
        R = np.loadtxt(os.path.join(root_dir, "train", "pose", database_pose_list[i]))
        # rotations.append(quaternion.from_rotation_matrix(R[:3, :3]))
        database_pose.append(R[:3, -1])

    for i in range(len(query_pose_list)):
        R = np.loadtxt(os.path.join(root_dir, "test", "pose", query_pose_list[i]))
        # rotations.append(quaternion.from_rotation_matrix(R[:3, :3]))
        query_pose.append(R[:3, -1])
    # database_pose = poses[:1000]
    # query_pose = poses[1000:]
    # print(query_pose)

    database_tree = KDTree(database_pose)

    database_sets = []
    test_sets = []
    database = {}
    for i in range(len(database_pose)):
        database[i] = {
            "query": os.path.join(
                rgb_dir, database_pose_list[i].replace("pose.txt", "color.png")
            )
        }
    database_sets.append(database)
    test = {}
    for i in range(len(query_pose)):
        test[i] = {
            "query": os.path.join(
                rgb_dir, query_pose_list[i].replace("pose.txt", "color.png")
            )
        }
    test_sets.append(test)
    for i in range(len(database_sets)):
        for j in range(len(test_sets)):
            for key in range(len(test_sets[j].keys())):
                print(key)
                ind_p = database_tree.query_radius([query_pose[key].tolist()], r=0.4)

                test_sets[j][key][i] = ind_p[0].tolist()
                # test_sets[j][key][i] = test_pickle[key]['positives']
            # for key in range(len(database_sets[i].keys())):

            #     database_sets[j][key][i] = train_pickle[key]['positives']
    print(test_sets)
    output_to_file(database_sets, root_dir, root_name + "_evaluation_database.pickle")
    output_to_file(test_sets, root_dir, root_name +"_evaluation_query.pickle")



import argparse
def main():
    parser = argparse.ArgumentParser(description="Generate Baseline training dataset")
    parser.add_argument(
        "--dataset_root", type=str, required=True, help="Dataset root folder"
    )
    args = parser.parse_args()
    print("Dataset root: {}".format(args.dataset_root))
    root_name = args.dataset_root
    construct_query_and_database_sets(root_name)

if __name__ == "__main__":
    main()

