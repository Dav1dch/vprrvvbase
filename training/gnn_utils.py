from datasets.augmentation import ValRGBTransform
import torch
import tqdm
import os
import copy
import random
import torch.nn as nn
import numpy as np
from models.model_factory import model_factory
from misc.utils import MinkLocParams
from datasets.augmentation import TrainTransform, TrainSetTransform, MinimizeTransform
from torch.utils.data import DataLoader, dataset
from torch.utils.data import Sampler
from datasets.seven_scenes import SevenScenesDatasets

from dgl.nn.pytorch import SAGEConv

# from training.gnn_sage import SAGEConv
import torch.nn.functional as F
import dgl.function as fn
from PIL import Image

# from sageconv_plus import SAGEConv_plus
from scipy.spatial import distance
from torch import autograd

train_sim_mat = []
query_sim_mat = []
database_sim_mat = []


def get_latent_vectors(model, set, device, params):
    # Adapted from original PointNetVLAD code

    model.eval()
    embeddings_l = []
    x = []
    f_maps = []
    for elem_ndx in tqdm.tqdm(set):
        x.append(
            load_data_item(
                set[elem_ndx]["query"],
                params,
            )
        )
        if len(x) == 10:
            x = torch.stack(x)

            with torch.no_grad():
                # coords are (n_clouds, num_points, channels) tensor
                batch = {}

                if params.use_rgb:
                    batch["images"] = x.to(device)

                x = model(batch)
                embedding = x["embedding"]

                # embedding is (1, 256) tensor
                if params.normalize_embeddings:
                    embedding = torch.nn.functional.normalize(
                        embedding, p=2, dim=1
                    )  # Normalize embeddings

            embedding = embedding.detach().cpu().numpy()
            embeddings_l.append(embedding)
            x = []

    embeddings = np.vstack(embeddings_l)
    return embeddings


def load_minkLoc_model(config, model_config, rgb_weights=None):
    rw = rgb_weights

    params = MinkLocParams(config, model_config)
    # params.print()

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    # print('Device: {}'.format(device))

    mink_model = model_factory(params)

    if rgb_weights is not None:
        assert os.path.exists(rgb_weights), "Cannot open network weights: {}".format(
            rgb_weights
        )
        # print('Loading weights: {}'.format(weights))
        mink_model.load_state_dict(
            torch.load(rgb_weights, map_location=device), strict=False
        )

    return mink_model, params


class MLP(nn.Module):
    def __init__(self, in_feats, out_feats) -> None:
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(in_feats, 256)
        self.linear2 = nn.Linear(256, 128)
        self.linear3 = nn.Linear(128, 64)
        self.linear4 = nn.Linear(64, 1)

    def forward(self, f):
        h = self.linear1(f)
        h = F.leaky_relu(h)
        h = self.linear2(h)
        h = F.leaky_relu(h)
        h = self.linear3(h)
        h = F.leaky_relu(h)
        h = self.linear4(h)
        # h = F.relu(h)
        h = torch.sigmoid(h)
        # h = F.relu(h)
        # return torch.sigmoid(h)
        return h


class myGNNCNN(nn.Module):
    def __init__(self, in_feats, h_feats, out_feats):
        super(myGNNCNN, self).__init__()

        self.MLP = MLP(in_feats, 1)
        self.BN = nn.BatchNorm1d(in_feats)
        self.conv1 = SAGEConv(in_feats, in_feats, "mean")
        self.conv_feature_map = nn.Sequential(
            # nn.BatchNorm3d(in_feats),
            nn.Conv3d(in_feats, in_feats, kernel_size=(2, 3, 3), stride=2),
            nn.ReLU(),
            nn.AdaptiveMaxPool3d((1, 1, 1)),
            nn.Flatten(1),
            nn.Linear(in_feats, in_feats),
            nn.Sigmoid(),
        )
        # self.conv2 = SAGEConv(in_feats, in_feats, "mean")

    def apply_edges_feature_map(self, edges):
        h_u = edges.src["x"]
        h_v = edges.dst["x"]
        # h_u = h_u.unsqueeze(2)
        # h_v = h_v.unsqueeze(2)
        h_u_v = torch.stack((h_u, h_v), dim=2)
        # score = self.MLP(torch.cat((h_u, h_v), 1))
        # score = self.MLP(h_u - h_v)
        score = self.conv_feature_map(h_u_v)
        return {"score": score}

    def apply_edges(self, edges):
        h_u = edges.src["x"]
        h_v = edges.dst["x"]
        # score = self.MLP(torch.cat((h_u, h_v), 1))
        score = self.MLP(h_u - h_v)
        # score = self.conv_feature_map(h_u_v)
        return {"score": score}

    def forward(self, g, x):
        # x = self.BN(x)

        with g.local_scope():
            g.ndata["x"] = x
            g.apply_edges(self.apply_edges_feature_map)
            e = g.edata["score"]

        A = self.conv1(g, x, e)
        A = F.leaky_relu(A)
        # A = self.conv2(g, A, e)
        # A = F.leaky_relu(A)
        A = F.normalize(A, dim=1)

        # with g.local_scope():
        #     g.ndata["x"] = A
        #     g.apply_edges(self.apply_edges)
        #     e = g.edata["score"]
        # pred2, A2 = self.conv2(g, pred)
        return A, e


class myGNN(nn.Module):
    def __init__(self, in_feats, h_feats, out_feats):
        super(myGNN, self).__init__()

        self.MLP = MLP(in_feats, 1)
        self.BN = nn.BatchNorm1d(in_feats)
        self.conv1 = SAGEConv(in_feats, in_feats, "mean")
        # self.conv2 = SAGEConv(in_feats, in_feats, "mean")

    def apply_edges(self, edges):
        h_u = edges.src["x"]
        h_v = edges.dst["x"]
        # score = self.MLP(torch.cat((h_u, h_v), 1))
        score = self.MLP(h_u - h_v)
        return {"score": score}

    def forward(self, g, x):
        # x = self.BN(x)

        with g.local_scope():
            g.ndata["x"] = x
            g.apply_edges(self.apply_edges)
            e = g.edata["score"]

        A = self.conv1(g, x, e)
        A = F.leaky_relu(A)
        # A = self.conv2(g, A, e)
        # A = F.leaky_relu(A)
        A = F.normalize(A, dim=1)
        # pred2, A2 = self.conv2(g, pred)
        return A, e


class ListDict(object):
    def __init__(self, items=None):
        if items is not None:
            self.items = copy.deepcopy(items)
            self.item_to_position = {item: ndx for ndx, item in enumerate(items)}
        else:
            self.items = []
            self.item_to_position = {}

    def add(self, item):
        if item in self.item_to_position:
            return
        self.items.append(item)
        self.item_to_position[item] = len(self.items) - 1

    def remove(self, item):
        position = self.item_to_position.pop(item)
        last_item = self.items.pop()
        if position != len(self.items):
            self.items[position] = last_item
            self.item_to_position[last_item] = position

    def choose_random(self):
        return random.choice(self.items)

    def __contains__(self, item):
        return item in self.item_to_position

    def __iter__(self):
        return iter(self.items)

    def __len__(self):
        return len(self.items)


class BatchSampler(Sampler):
    # Sampler returning list of indices to form a mini-batch
    # Samples elements in groups consisting of k=2 similar elements (positives)
    # Batch has the following structure: item1_1, ..., item1_k, item2_1, ... item2_k, itemn_1, ..., itemn_k
    def __init__(self, dataset: SevenScenesDatasets, batch_size: int, type: str):
        self.batch_size = batch_size
        self.max_batches = batch_size
        self.dataset = dataset
        self.type = type
        self.k = 2

        # Index of elements in each batch (re-generated every epoch)
        self.batch_idx = []
        # List of point cloud indexes
        self.elems_ndx = list(self.dataset.queries)

    def __iter__(self):
        # Re-generate batches every epoch
        if self.type == "train":
            # self.generate_top_batches()
            self.generate_smoothap_batches()
            # self.generate_smoothap_batches()
        else:
            self.generate_smoothap_val_batches()

        for batch in self.batch_idx:
            yield batch

    def __len__(self):
        return len(self.batch_idx)

    def generate_smoothap_batches(self):
        self.batch_idx = []
        for ndx in range(len(self.dataset)):
            current_batch = []
            current_batch.append(ndx)
            self.batch_idx.append(current_batch)
        random.shuffle(self.batch_idx)

    def generate_smoothap_val_batches(self):
        self.batch_idx = []
        for ndx in range(len(self.dataset)):
            current_batch = []
            current_batch.append(ndx)
            self.batch_idx.append(current_batch)


def in_sorted_array(e: int, array: np.ndarray) -> bool:
    # if array == None or len(array) == 0:
    if len(array) == 0:
        return False
    array = np.sort(array)
    pos = np.searchsorted(array, e)
    if pos == len(array) or pos == -1:
        return False
    else:
        return bool(array[pos] == e)
        # return True


def make_smoothap_collate_fn(
    dataset: SevenScenesDatasets, mink_quantization_size=None, val=None
):
    # set_transform: the transform to be applied to all batch elements
    def collate_fn(data_list):
        # Constructs a batch object
        global train_sim_mat
        global database_sim_mat
        global query_sim_mat
        num = 50
        positives_mask = []
        negatives_mask = []
        labels = [e["ndx"] for e in data_list]
        if val != "val":
            labels.extend(train_sim_mat[labels[0]][:num])
            positives_mask = [
                in_sorted_array(e, dataset.queries[labels[0]].positives) for e in labels
            ]
            positives_mask[0] = True
            negatives_mask = [not item for item in positives_mask]
            positives_mask = torch.tensor([positives_mask])
            negatives_mask = torch.tensor([negatives_mask])
        else:

            labels.extend(query_sim_mat[labels[0]][:num])
            positives_mask = [
                in_sorted_array(e, dataset.queries[labels[0]].positives) for e in labels
            ]
            # hard_positives_mask = [in_sorted_array(
            #     e, dataset.queries[labels[0]].hard_positives) for e in labels]
            positives_mask[0] = True
            negatives_mask = [not item for item in positives_mask]
            positives_mask = torch.tensor([positives_mask])
            negatives_mask = torch.tensor([negatives_mask])

        neighbours = []
        if val == "val":
            neighbours.append(query_sim_mat[labels[0]][:num])
            neighbours_temp = [database_sim_mat[item][:num] for item in labels[1:]]
            neighbours.extend(neighbours_temp)

        else:
            # neighbours = [dataset.get_neighbours(item)[:10] for item in labels]
            for i in labels:
                temp = train_sim_mat[i][1 : num + 1]

                neighbours.append(temp)
        return (
            positives_mask,
            negatives_mask,
            labels,
            neighbours,
            None,
        )

    return collate_fn


def make_dataloader(params):
    datasets = {}
    # dataset_folder = "/home/david/datasets/fire"

    train_transform = TrainTransform(1)
    train_set_transform = TrainSetTransform(1)

    train_embeddings = np.load("./gnn_pre_train_embeddings.npy")
    test_embeddings = np.load("./gnn_pre_test_embeddings.npy")
    database_embeddings = train_embeddings
    query_embeddings = test_embeddings
    global train_sim_mat
    global database_sim_mat
    global query_sim_mat

    train_sim = distance.cdist(train_embeddings, train_embeddings)
    database_sim = distance.cdist(database_embeddings, database_embeddings)
    query_sim = distance.cdist(query_embeddings, database_embeddings)

    # train_sim = np.matmul(train_embeddings, train_embeddings.T)
    # database_sim = np.matmul(test_embeddings, test_embeddings.T)
    # train_sim_mat = np.argsort(train_sim)
    # database_sim_mat = np.argsort(database_sim)

    train_sim_mat = np.argsort(train_sim).tolist()
    database_sim_mat = np.argsort(database_sim).tolist()
    query_sim_mat = np.argsort(query_sim).tolist()

    t_ = []
    for i in range(len(train_sim_mat)):
        t = np.array(train_sim_mat[i])
        t = t[t != i]
        t = t[(t != i) & (t // 500 != i // 500)]
        # ind = np.random.randint(120, size=55)
        # t = t[ind]

        t_.append(list(t))
    train_sim_mat = t_

    # database_sim_mat = train_sim_mat.copy()

    datasets["train"] = SevenScenesDatasets(params.dataset_folder, params.train_file)
    datasets["val"] = SevenScenesDatasets(params.dataset_folder, params.val_file)
    val_transform = None

    dataloaders = {}
    train_sampler = BatchSampler(datasets["train"], batch_size=100, type="train")
    # Collate function collates items into a batch and applies a 'set transform' on the entire batch
    train_collate_fn = make_smoothap_collate_fn(datasets["train"], 0.01)
    dataloaders["train"] = DataLoader(
        datasets["train"],
        batch_sampler=train_sampler,
        collate_fn=train_collate_fn,
        num_workers=params.num_workers,
        pin_memory=False,
    )

    if "val" in datasets:
        val_sampler = BatchSampler(datasets["val"], batch_size=100, type="val")
        # Collate function collates items into a batch and applies a 'set transform' on the entire batch
        # Currently validation dataset has empty set_transform function, but it may change in the future
        val_collate_fn = make_smoothap_collate_fn(datasets["val"], 0.01, "val")
        dataloaders["val"] = DataLoader(
            datasets["val"],
            batch_sampler=val_sampler,
            collate_fn=val_collate_fn,
            num_workers=params.num_workers,
            pin_memory=True,
        )
    return dataloaders


def load_data_item(file_name, params):
    # returns Nx3 matrix
    file_path = os.path.join(params.dataset_folder, file_name)

    result = {}

    img = Image.open(file_path)
    transform = MinimizeTransform()
    # result["image"] = transform(img)
    result = transform(img)

    return result
