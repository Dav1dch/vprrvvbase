import tqdm
import torch
import os
import copy
import random
import torch.nn as nn
import numpy as np
from models.model_factory import model_factory
from misc.utils import MinkLocParams
from datasets.seven_scenes import TrainTransform, TrainSetTransform
from torch.utils.data import DataLoader
from torch.utils.data import Sampler
from datasets.seven_scenes import SevenScenesDatasets
import MinkowskiEngine as ME
from dgl.nn.pytorch import SAGEConv
import torch.nn.functional as F
import dgl.function as fn

# from sageconv_plus import SAGEConv_plus
import gc

train_sim_mat = None
query_sim_mat = None
database_sim_mat = None


class MLPModel(torch.nn.Module):

    def __init__(self, num_i, num_h, num_o):
        super(MLPModel, self).__init__()

        self.BN = nn.BatchNorm1d(256)
        self.linear1 = torch.nn.Linear(num_i, num_h)
        self.relu = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(num_h, num_h)  # 2个隐层
        self.relu2 = torch.nn.ReLU()
        self.linear3 = torch.nn.Linear(num_h, num_o)

    def forward(self, x):
        x = self.BN(x)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)

        x = F.normalize(x, dim=1)
        return x


def load_minkLoc_model(config, model_config, weights=None):
    w = weights
    # print('Weights: {}'.format(w))
    # print('')

    params = MinkLocParams(config, model_config)
    # params.print()

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    # print('Device: {}'.format(device))

    mink_model = model_factory(params)
    if weights is not None:
        assert os.path.exists(weights), "Cannot open network weights: {}".format(
            weights
        )
        # print('Loading weights: {}'.format(weights))
        mink_model.load_state_dict(torch.load(weights, map_location=device))

    return mink_model, params


def load_pc(file_name):
    # returns Nx3 matrix
    pc = np.fromfile(file_name, dtype=np.float32)
    # coords are within -1..1 range in each dimension
    # assert pc.shape[0] == params.num_points * 3, "Error in point cloud shape: {}".format(file_path)
    pc = np.reshape(pc, (pc.shape[0] // 3, 3))
    pc = torch.tensor(pc, dtype=torch.float)
    return pc


class DotPruductPredictor(nn.Module):
    def forward(self, graph, h):
        with graph.local_scope():
            graph.ndata["h"] = h
            graph.apply_edges(fn.u_dot_v("h", "h", "score"))
            return graph.edata["score"]


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


class Edge_regression(nn.Module):
    def __init__(self):
        super(Edge_regression, self).__init__()
        # self.conv = SAGEConv_plus(in_feats, out_feats, aggregate)
        self.MLP = MLP(512, 1)
        # self.linear1 = nn.Linear(256, 512)
        # self.linear2 = nn.Linear(512, 256)

    def apply_edges(self, edges):
        h_u = edges.src["x"]
        h_v = edges.dst["x"]
        score = self.MLP(torch.cat((h_u, h_v), 1))
        # score = self.MLP(h_u - h_v)
        return {"score": score}

    def forward(self, g, x):
        with g.local_scope():
            g.ndata["x"] = x
            g.apply_edges(self.apply_edges)
            e = g.edata["score"]
            return g.edata["score"]


class SAGE_Conv_layer(nn.Module):
    def __init__(self, in_feats, out_feats, aggregate):
        super(SAGE_Conv_layer, self).__init__()
        self.conv = SAGEConv_plus(in_feats, out_feats, aggregate)

    def forward(self, g, x):

        h = self.conv(g, x)
        h = F.leaky_relu(h)

        h = F.normalize(h, p=2, dim=1)
        return h


class myGNN(nn.Module):
    def __init__(self, in_feats, h_feats, out_feats):
        super(myGNN, self).__init__()

        self.MLP = MLP(256, 1)
        # self.BN = nn.BatchNorm1d(256)
        self.conv1 = SAGEConv(in_feats, 256, "mean")

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
        # self.generate_batches()
        # self.generate_most_only_batches()
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

    def generate_batches(self):
        # Generate training/evaluation batches.
        # batch_idx holds indexes of elements in each batch as a list of lists
        self.batch_idx = []

        unused_elements_ndx = ListDict(self.elems_ndx)
        current_batch = []

        assert self.k == 2, "sampler can sample only k=2 elements from the same class"

        while True:
            if len(current_batch) >= self.batch_size or len(unused_elements_ndx) == 0:
                # Flush out batch, when it has a desired size, or a smaller batch, when there's no more
                # elements to process
                if len(current_batch) >= 2 * self.k:
                    # Ensure there're at least two groups of similar elements, otherwise, it would not be possible
                    # to find negative examples in the batch
                    assert (
                        len(current_batch) % self.k == 0
                    ), "Incorrect bach size: {}".format(len(current_batch))
                    self.batch_idx.append(current_batch)
                    current_batch = []
                    if (self.max_batches is not None) and (
                        len(self.batch_idx) >= self.max_batches
                    ):
                        break
                if len(unused_elements_ndx) == 0:
                    break

            # Add k=2 similar elements to the batch
            selected_element = unused_elements_ndx.choose_random()
            unused_elements_ndx.remove(selected_element)
            positives = self.dataset.get_positives(selected_element)
            if len(positives) == 0:
                # Broken dataset element without any positives
                continue

            unused_positives = [e for e in positives if e in unused_elements_ndx]
            # If there're unused elements similar to selected_element, sample from them
            # otherwise sample from all similar elements
            if len(unused_positives) > 0:
                second_positive = random.choice(unused_positives)
                unused_elements_ndx.remove(second_positive)
            else:
                second_positive = random.choice(list(positives))

            current_batch += [selected_element, second_positive]

        for batch in self.batch_idx:
            assert len(batch) % self.k == 0, "Incorrect bach size: {}".format(
                len(batch)
            )


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
        num = 45
        positives_mask = []
        negatives_mask = []
        hard_positives_mask = []
        most_positives_mask = []
        labels = [e["ndx"] for e in data_list]
        if val != "val":
            labels.extend(train_sim_mat[labels[0]][1:51])
            positives_mask = [
                in_sorted_array(e, dataset.queries[labels[0]].positives) for e in labels
            ]
            # hard_positives_mask = [
            #     in_sorted_array(e, dataset.queries[labels[0]].hard_positives)
            #     for e in labels
            # ]
            positives_mask[0] = True
            negatives_mask = [not item for item in positives_mask]
            positives_mask = torch.tensor([positives_mask])
            negatives_mask = torch.tensor([negatives_mask])
            # most_positives_mask = [
            #     in_sorted_array(e, dataset.queries[labels[0]].most_positive)
            #     for e in labels
            # ]
            # most_positives_mask = torch.tensor([most_positives_mask])
            # hard_positives_mask = torch.tensor([hard_positives_mask])
        else:
            labels.extend(
                query_sim_mat[labels[0]][: min(50, len(query_sim_mat[labels[0]]))]
            )

            positives_mask = [
                in_sorted_array(e, dataset.queries[labels[0]].positives) for e in labels
            ]
            # hard_positives_mask = [in_sorted_array(
            #     e, dataset.queries[labels[0]].hard_positives) for e in labels]
            positives_mask[0] = True
            negatives_mask = [not item for item in positives_mask]
            positives_mask = torch.tensor([positives_mask])
            negatives_mask = torch.tensor([negatives_mask])
            # hard_positives_mask = torch.tensor([hard_positives_mask])
            # most_positives_mask = [
            #     in_sorted_array(e, dataset.queries[labels[0]].most_positive)
            #     for e in labels
            # ]
            # most_positives_mask = torch.tensor([most_positives_mask])

        neighbours = []
        if val == "val":
            neighbours.append(
                query_sim_mat[labels[0]][: min(50, len(query_sim_mat[labels[0]]))]
            )
            neighbours_temp = [
                database_sim_mat[item][: min(50, len(database_sim_mat[item]))]
                for item in labels[1:]
            ]
            neighbours.extend(neighbours_temp)

        else:
            # neighbours = [dataset.get_neighbours(item)[:10] for item in labels]
            for i in labels:
                temp = train_sim_mat[i][1:51]

                neighbours.append(temp)
        return (
            positives_mask,
            negatives_mask,
            # hard_positives_mask,
            labels,
            neighbours,
            # most_positives_mask,
            None,
        )

    return collate_fn


def make_dataloader(params):
    datasets = {}
    dataset_folder = "/home/david/datasets/kitti/"
    train_file = "kitti_train_dist.pickle"
    test_file = "kitti_test_dist.pickle"

    root_dir = "/home/david/datasets/kitti/"

    seqs = ["03", "04", "05", "06", "07", "08", "09", "10"]

    seq_num = []

    # for s in seqs:
    #     num = len(os.listdir(os.path.join(root_dir, "sequences", s, "velodyne")))
    #     seq_num.append(num)

    for s in seqs:
        # num = len(os.listdir(os.path.join(root_dir, "sequences", s, "velodyne")))
        num = len(np.loadtxt(os.path.join(root_dir, "poses", s + ".txt")))
        seq_num.append(num)
    train_cum_sum = np.cumsum(seq_num).tolist()

    train_transform = TrainTransform(1)
    train_set_transform = TrainSetTransform(1)
    train_embeddings = np.load("./gnn_pre_train_embeddings.npy")
    test_embeddings = np.load("./gnn_pre_test_embeddings.npy")
    global train_sim_mat
    global database_sim_mat
    global query_sim_mat

    train_sim = np.matmul(train_embeddings, train_embeddings.T)
    database_sim = np.matmul(test_embeddings, test_embeddings.T)
    train_sim_mat = np.argsort(-train_sim)
    database_sim_mat = np.argsort(-database_sim)
    del train_embeddings, test_embeddings
    gc.collect()
    import tqdm

    t_ = []
    for i in tqdm.tqdm(range(len(train_sim_mat))):

        seq_ind = 0
        for ind in range(len(train_cum_sum)):
            if ind < train_cum_sum[ind]:
                seq_ind = ind
                break

        if seq_ind == 0:
            start = 0
            end = train_cum_sum[seq_ind]
        else:
            start = train_cum_sum[seq_ind - 1]
            end = train_cum_sum[seq_ind]

        to_remove = []
        # for j in range(len(train_sim_mat)):
        t = np.array(train_sim_mat[i])
        t_.append(t[(t < start) | (t >= end)])

        # for j in train_sim_mat[i]:
        #     if j >= start and j < end:
        #         to_remove.append(j)
        # for j in to_remove:
        #     train_sim_mat[i].remove(j)
    train_sim_mat = t_

    t_ = []
    for i in tqdm.tqdm(range(len(database_sim_mat))):
        t = np.array(database_sim_mat[i])
        t_.append(t[(abs(t - i) > 100) & (t < i)])

        # to_remove = []
        # for j in database_sim_mat[i]:
        #     if abs(j - i) < 100 or j >= i:
        #         to_remove.append(j)
        # for j in to_remove:
        #     database_sim_mat[i].remove(j)
    database_sim_mat = t_

    query_sim_mat = database_sim_mat

    datasets["train"] = SevenScenesDatasets(dataset_folder, train_file, train_transform)
    datasets["val"] = SevenScenesDatasets(dataset_folder, test_file, None)

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


def get_embeddings(mink_model, params, set):
    # file_path = "/home/david/datasets/kitti/color"
    # file_li = os.listdir(file_path)
    # file_li.sort()
    embeddings_l = []
    mink_model.eval()
    for elem in tqdm.tqdm(set):
        x = load_data_item(os.path.join(set[elem].rel_scan_filepath), params)
        x = {"images": x.unsqueeze(0).cuda()}

        # coords are (n_clouds, num_points, channels) tensor
        with torch.no_grad():

            embedding = mink_model(x)["image_embedding"]
            # embedding is (1, 1024) tensor
            if params.normalize_embeddings:
                embedding = torch.nn.functional.normalize(
                    embedding, p=2, dim=1
                )  # Normalize embeddings

        embedding = embedding.detach().cpu().numpy()
        embeddings_l.append(embedding)

    embeddings = np.vstack(embeddings_l)
    return embeddings


from PIL import Image
from datasets.seven_scenes import MinimizeTransform


def load_data_item(file_name, params):
    # returns Nx3 matrix
    file_path = os.path.join(params.dataset_folder, file_name)

    result = {}

    img = Image.open(file_path)
    transform = MinimizeTransform()
    # result["image"] = transform(img)
    result = transform(img)

    return result
