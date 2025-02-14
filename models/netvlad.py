import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from sklearn.neighbors import NearestNeighbors
import numpy as np
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.utils.data.dataset import Subset
import h5py
import faiss
from math import log10, ceil
import random, shutil, json
from os.path import join, exists, isfile, realpath, dirname
from os import makedirs, remove, chdir, environ


# based on https://github.com/lyakaap/NetVLAD-pytorch/blob/master/netvlad.py
class NetVLAD(nn.Module):
    """NetVLAD layer implementation"""

    def __init__(self, num_clusters=64, dim=128, normalize_input=True, vladv2=False):
        """
        Args:
            num_clusters : int
                The number of clusters
            dim : int
                Dimension of descriptors
            alpha : float
                Parameter of initialization. Larger value is harder assignment.
            normalize_input : bool
                If true, descriptor-wise L2 normalization is applied to input.
            vladv2 : bool
                If true, use vladv2 otherwise use vladv1
        """
        super(NetVLAD, self).__init__()
        self.num_clusters = num_clusters
        self.dim = dim
        self.alpha = 0
        self.vladv2 = vladv2
        self.normalize_input = normalize_input
        self.conv = nn.Conv2d(dim, num_clusters, kernel_size=(1, 1), bias=vladv2)
        self.centroids = nn.Parameter(torch.rand(num_clusters, dim))

    def init_params(self, clsts, traindescs):
        # TODO replace numpy ops with pytorch ops
        if self.vladv2 == False:
            clstsAssign = clsts / np.linalg.norm(clsts, axis=1, keepdims=True)
            dots = np.dot(clstsAssign, traindescs.T)
            dots.sort(0)
            dots = dots[::-1, :]  # sort, descending

            self.alpha = (-np.log(0.01) / np.mean(dots[0, :] - dots[1, :])).item()
            self.centroids = nn.Parameter(torch.from_numpy(clsts))
            self.conv.weight = nn.Parameter(
                torch.from_numpy(self.alpha * clstsAssign).unsqueeze(2).unsqueeze(3)
            )
            self.conv.bias = None
        else:
            knn = NearestNeighbors(n_jobs=-1)  # TODO faiss?
            knn.fit(traindescs)
            del traindescs
            dsSq = np.square(knn.kneighbors(clsts, 2)[1])
            del knn
            self.alpha = (-np.log(0.01) / np.mean(dsSq[:, 1] - dsSq[:, 0])).item()
            self.centroids = nn.Parameter(torch.from_numpy(clsts))
            del clsts, dsSq

            self.conv.weight = nn.Parameter(
                (2.0 * self.alpha * self.centroids).unsqueeze(-1).unsqueeze(-1)
            )
            self.conv.bias = nn.Parameter(-self.alpha * self.centroids.norm(dim=1))

    def forward(self, x):
        N, C = x.shape[:2]

        if self.normalize_input:
            x = F.normalize(x, p=2, dim=1)  # across descriptor dim

        # soft-assignment
        soft_assign = self.conv(x).view(N, self.num_clusters, -1)
        soft_assign = F.softmax(soft_assign, dim=1)

        x_flatten = x.view(N, C, -1)

        # calculate residuals to each clusters
        vlad = torch.zeros(
            [N, self.num_clusters, C], dtype=x.dtype, layout=x.layout, device=x.device
        )
        for C in range(
            self.num_clusters
        ):  # slower than non-looped, but lower memory usage
            residual = x_flatten.unsqueeze(0).permute(1, 0, 2, 3) - self.centroids[
                C : C + 1, :
            ].expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)
            residual *= soft_assign[:, C : C + 1, :].unsqueeze(2)
            vlad[:, C : C + 1, :] = residual.sum(dim=-1)

        vlad = F.normalize(vlad, p=2, dim=2)  # intra-normalization
        vlad = vlad.view(x.size(0), -1)  # flatten
        vlad = F.normalize(vlad, p=2, dim=1)  # L2 normalize

        return vlad


class L2Norm(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, input):
        return F.normalize(input, p=2, dim=self.dim)


class NetVLADVGG16(nn.Module):
    """NetVLAD layer implementation"""

    def __init__(self, num_clusters=64, dim=512, normalize_input=True, vladv2=False):
        """
        Args:
            num_clusters : int
                The number of clusters
            dim : int
                Dimension of descriptors
            alpha : float
                Parameter of initialization. Larger value is harder assignment.
            normalize_input : bool
                If true, descriptor-wise L2 normalization is applied to input.
            vladv2 : bool
                If true, use vladv2 otherwise use vladv1
        """
        super(NetVLADVGG16, self).__init__()
        encoder = models.vgg16(pretrained=True)
        layers = list(encoder.features.children())[:-2]
        # if using pretrained then only train conv5_1, conv5_2, and conv5_3
        for l in layers[:-5]:
            for p in l.parameters():
                p.requires_grad = False
        self.layers = layers
        self.model = nn.Module()

    def init_cluster(self, cluster_set):
        layers = self.layers
        layers.append(L2Norm())
        encoder = nn.Sequential(*layers)
        model = nn.Module()
        model.add_module("encoder", encoder)
        model.to("cuda")
        get_clusters(cluster_set, model)

    def construct_model(self):
        initcache = join("centroids", "vgg16_fire_desc_cen.hdf5")
        net_vlad = NetVLAD(num_clusters=32, dim=512)
        with h5py.File(initcache, mode="r") as h5:
            clsts = h5.get("centroids")[...]
            traindescs = h5.get("descriptors")[...]
            net_vlad.init_params(clsts, traindescs)
            del clsts, traindescs
        self.model.add_module("encoder", nn.Sequential(*self.layers))
        self.model.add_module("vlad", net_vlad)
        self.model.to("cuda")

    def forward(self, batch):
        x = batch["images"]
        embedding = self.model.encoder(x)
        vlad = self.model.vlad(embedding)

        return vlad


def get_clusters(cluster_set, model):
    nDescriptors = 50000
    nPerImage = 100
    nIm = ceil(nDescriptors / nPerImage)

    sampler = SubsetRandomSampler(
        np.random.choice(len(cluster_set), nIm, replace=False)
    )
    data_loader = DataLoader(
        dataset=cluster_set,
        num_workers=8,
        batch_size=24,
        shuffle=False,
        sampler=sampler,
    )

    if not exists(join("centroids")):
        makedirs(join("centroids"))

    initcache = join(
        "centroids",
        "vgg16_" + "fire" + "_desc_cen.hdf5",
    )
    with h5py.File(initcache, mode="w") as h5:
        with torch.no_grad():
            model.eval()
            print("====> Extracting Descriptors")
            dbFeat = h5.create_dataset(
                "descriptors", [nDescriptors, 512], dtype=np.float32
            )

            for iteration, input in enumerate(data_loader, 1):
                input = input["image"].to("cuda")
                image_descriptors = (
                    model.encoder(input).view(input.size(0), 512, -1).permute(0, 2, 1)
                )

                batchix = (iteration - 1) * 24 * nPerImage
                for ix in range(image_descriptors.size(0)):
                    # sample different location for each image in batch
                    sample = np.random.choice(
                        image_descriptors.size(1), nPerImage, replace=False
                    )
                    startix = batchix + ix * nPerImage
                    dbFeat[startix : startix + nPerImage, :] = (
                        image_descriptors[ix, sample, :].detach().cpu().numpy()
                    )

                if iteration % 50 == 0 or len(data_loader) <= 10:
                    print(
                        "==> Batch ({}/{})".format(iteration, ceil(nIm / 24)),
                        flush=True,
                    )
                del input, image_descriptors

        print("====> Clustering..")
        niter = 100
        kmeans = faiss.Kmeans(512, 32, niter=niter, verbose=False)
        kmeans.train(dbFeat[...])

        print("====> Storing centroids", kmeans.centroids.shape)
        h5.create_dataset("centroids", data=kmeans.centroids)
        print("====> Done!")
