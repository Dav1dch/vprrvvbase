import gc
from eval.evaluate import evaluate_dataset
import tqdm
import numpy as np
import torch
import torch.nn as nn
import pickle
import dgl
from gnn_loss import ERFA, SmoothAP

# import warnings
# warnings.filterwarnings("ignore")


from gnn_utils_kitti import (
    load_minkLoc_model,
    make_dataloader,
    myGNN,
    get_embeddings,
)


config = "/home/david/Code/vprvv-base/config/config_baseline_rgb.txt"
model_config = "/home/david/Code/vprvv-base/models/minklocrgb.txt"
weights = (
    "/home/david/Code/vprvv-base/weights/model_MinkLocRGB_20240326_1932_kitti_best.pth"
)


if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print("Device: {}".format(device))


# # load minkloc
mink_model, params = load_minkLoc_model(config, model_config, weights)
mink_model.to(device)
print(params.train_file)
print(params.val_file)
# import pickle
#
with open("/home/david/datasets/kitti/" + params.train_file, "rb") as f:
    p = pickle.load(f)
train_embeddings = get_embeddings(mink_model, params, p)
#
#
with open("/home/david/datasets/kitti/" + params.val_file, "rb") as f:
    p = pickle.load(f)
test_embeddings = get_embeddings(mink_model, params, p)
#
# # train_embeddings = embeddings[9202:]
# # test_embeddings = embeddings[:4541]
# # del embeddings
# print(train_embeddings.shape)
# print(test_embeddings.shape)
# #
# # gc.collect()
np.save("./gnn_pre_train_embeddings.npy", train_embeddings)
np.save("./gnn_pre_test_embeddings.npy", test_embeddings)


# test_embeddings = get_embeddings(mink_model, params, device, 'test')

train_embeddings = np.load("./gnn_pre_train_embeddings.npy")
test_embeddings = np.load("./gnn_pre_test_embeddings.npy")

train_embeddings = torch.tensor(train_embeddings).to("cuda")
test_embeddings = torch.tensor(test_embeddings).to("cuda")


# database_embeddings = torch.tensor(test_embeddings[:1000]).to('cuda')
# query_embeddings = torch.tensor(test_embeddings[1000:]).to('cuda')

# load dataloaders
dataloaders = make_dataloader(params)


# load evaluate file
with open("/home/david/datasets/kitti/kitti_evaluation_database.pickle", "rb") as f:
    # with open('/home/david/datasets/apt1/kitchen/pickle/apt1_evaluation_database.pickle', 'rb') as f:
    database = pickle.load(f)

with open("/home/david/datasets/kitti/kitti_evaluation_query.pickle", "rb") as f:
    # with open('/home/david/datasets/apt1/kitchen/pickle/apt1_evaluation_query.pickle', 'rb') as f:
    query = pickle.load(f)


# load training pickle
train_pickle = pickle.load(
    open("/home/david/datasets/kitti/kitti_train_dist.pickle", "rb")
)
# test_pickle = pickle.load(open('/home/david/datasets/pumpkin/pickle/pumpkin_test_overlap.pickle', 'rb'))

# load iou file
train_iou = np.load("/home/david/Code/S3E_PreProcess/kitti_iou.npy")
# test_iou = np.load('/home/david/datasets/pumpkin/iou_heatmap/test_iou_heatmap.npy')


gt = [[0.001 for _ in range(len(train_iou))] for _ in range(len(train_iou))]
# test_gt = [[0. for _ in range(len(test_iou))] for _ in range(len(test_iou))]

gt = torch.tensor(gt)
# test_gt = torch.tensor(test_gt)

for i in train_pickle:
    gt[i][i] = 1.0
    for p in train_pickle[i].positives:
        gt[i][p] = 1.0
np.save("./gt_mat.npy", gt.numpy())

# for i in test_iou:
#     # test_gt[i][i] = 1.
#     for p in test_pickle[i].positives:
#         test_gt[i][p] = 1.

# np.save('test_gt.npy',test_gt.numpy())

model = myGNN(256, 512, 256)
model.to("cuda")

# mlpmodel = MLPModel(256, 512, 256)
# mlpmodel.to('cuda')

opt = torch.optim.Adam(
    [{"params": model.parameters(), "lr": 0.001, "weight_decay": 0.0001}]
)

loss = None
recall = None


smoothap = ERFA()
# caliloss = Cali_loss()
d = {"loss": loss}


criterion = nn.MSELoss().to("cuda")
# shrinkage_loss = Shrinkage_loss(5, 0.2).to('cuda')
pdist = nn.PairwiseDistance(p=2)
cos = nn.CosineSimilarity(dim=1).cuda()

max_ = 0.0
# labels = range(len(feat))
with tqdm.tqdm(range(100), position=0, desc="epoch", ncols=60) as tbar:
    for i in tbar:
        # loss status
        loss = 0.0
        losses = []
        cnt = 0.0
        num_evaluated = 0.0
        recall = [0] * 50
        opt.zero_grad()

        with tqdm.tqdm(
            dataloaders["train"], position=1, desc="batch", ncols=60
        ) as tbar2:
            for (
                pos_mask,
                neg_mask,
                # hard_pos_mask,
                labels,
                neighbours,
                # most_pos_mask,
                batch,
            ) in tbar2:
                torch.cuda.empty_cache()
                cnt += 1
                model.train()
                # mlpmodel.train()

                with torch.enable_grad():
                    # batch = {e: batch[e].to(device) for e in batch}
                    src = np.array(list(range(len(labels))))
                    dst = np.repeat([0], len(labels))
                    src, dst = src + dst, dst + src

                    # src = np.array(
                    #     list(range(1, len(labels) * (len(labels) - 1) + 1)))
                    # dst = np.repeat(list(range(len(labels))), len(labels) - 1)
                    #
                    g = dgl.graph((src, dst))
                    g = g.to("cuda")
                    # ind = [labels[0]]
                    ind = labels
                    # ind.extend(np.vstack(neighbours).reshape((-1,)).tolist())
                    # indx = torch.tensor(ind).view((-1,))[dst[:len(labels) - 1]]
                    # indy = torch.tensor(ind)[src[:len(labels) - 1]]
                    indx = torch.tensor(ind).view((-1,))[dst[:]]
                    indy = torch.tensor(ind)[src[:]]
                    embeddings = train_embeddings[ind]
                    gt_iou = gt[indx, indy].view((-1, 1))
                    A, e = model(g, embeddings)
                    # A = mlpmodel(embeddings)
                    # query_embs = A[0].unsqueeze(0)
                    query_embs = torch.repeat_interleave(
                        A[0].unsqueeze(0), len(labels) - 1, 0
                    )
                    database_embs = A[1 : len(labels)]
                    sim_mat = cos(query_embs, database_embs)
                    # sim_mat = torch.matmul(query_embs, database_embs.T).squeeze()

                    loss_affinity_1 = criterion(e[: len(labels)], gt_iou.cuda())

                    # hard_sim_mat = sim_mat[pos_mask.squeeze()[1:]]
                    #
                    # hard_pos_mask[0][0] = True
                    # hard_p_mask = hard_pos_mask[pos_mask].unsqueeze(0)
                    # ap_coarse = smoothap(sim_mat, pos_mask)
                    # ap_fine = smoothap(hard_sim_mat, hard_p_mask)

                    losses.append(
                        1 - smoothap(sim_mat, pos_mask, gt_iou) + loss_affinity_1
                    )
                    # losses.append(
                    #     1 - (0.9 * ap_coarse + 0.1 * ap_fine) + loss_affinity_1
                    # )

                    loss += losses[-1].item()
                    if cnt % 256 == 0 or cnt == len(train_embeddings):
                        a = torch.vstack(losses)
                        a = torch.where(torch.isnan(a), torch.full_like(a, 0), a)
                        loss_smoothap = torch.mean(a)
                        loss_smoothap.backward()
                        opt.step()
                        opt.zero_grad()
                        losses = []

                    rank = np.argsort(-sim_mat.detach().cpu().numpy())
                    # true_neighbors = database[0][labels[0]][0]
                    true_neighbors = train_pickle[labels[0]].positives
                    if len(true_neighbors) == 0:
                        continue
                    num_evaluated += 1

                    flag = 0
                    for j in range(len(rank)):
                        # if rank[j] == 0:
                        #     flag = 1
                        #     continue
                        if labels[1:][rank[j]] in true_neighbors:
                            # if j == 0:
                            #     similarity = sim_mat[rank[j]]
                            #     top1_similarity_score.append(similarity)
                            recall[j - flag] += 1
                            break

                # tbar2.set_postfix({'loss' :loss_smoothap.item()})

            # print(loss / cnt)
            # recall = (np.cumsum(recall)/float(num_evaluated))*100
            print("train recall\n", recall[0])

            # print(pos_mask)
            # print(gt_iou.view(-1,)[:len(pos_mask[0])])

            t_loss = 0.0

            with torch.no_grad():
                recall = [0] * 50
                num_evaluated = 0
                top1_similarity_score = []
                one_percent_retrieved = 0
                threshold = max(int(round(2000 / 100.0)), 1)

                starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
                    enable_timing=True
                )
                timings = []
                # timings = np.zeros((3000, 1))

                t_loss = 0.0

                # src = np.array(
                #     list(range(1, 51 * (51 - 1) + 1)))
                # dst = np.repeat(
                #     list(range(51)), 51 - 1)
                #
                # g = dgl.graph((src, dst))
                # g = g.to('cuda')
                with tqdm.tqdm(
                    dataloaders["val"], position=1, desc="batch", ncols=50
                ) as tbar3:
                    for (
                        pos_mask,
                        neg_mask,
                        # hard_pos_mask,
                        labels,
                        neighbours,
                        # most_pos_mask,
                        batch,
                    ) in tbar3:
                        model.eval()
                        true_neighbors = query[0][labels[0]][0]

                        if len(true_neighbors) == 0:
                            continue
                        print(true_neighbors)

                        src = np.array(list(range(len(labels))))
                        dst = np.repeat([0], len(labels))
                        src, dst = src + dst, dst + src

                        # src = np.array(
                        #     list(range(1, len(labels) * (len(labels) - 1) + 1)))
                        # dst = np.repeat(
                        #     list(range(len(labels))), len(labels) - 1)
                        g = dgl.graph((src, dst))
                        g = g.to("cuda")
                        # ind = [labels[0]]
                        ind = labels
                        # ind.extend(
                        #     np.vstack(neighbours).reshape((-1,)).tolist())
                        embeddings = torch.vstack(
                            (test_embeddings[ind[0]], test_embeddings[ind[1:]])
                        ).cuda()
                        starter.record()
                        A, e = model(g, embeddings)
                        ender.record()
                        torch.cuda.synchronize()  # 等待GPU任务完成

                        timings.append(starter.elapsed_time(ender))
                        # A = mlpmodel(embeddings)
                        database_embs = A[1 : len(labels)]
                        q = A[0].unsqueeze(0)
                        query_embs = torch.repeat_interleave(q, len(labels) - 1, 0)
                        # sim_mat = torch.matmul(q, database_embs.T).squeeze()

                        sim_mat = cos(query_embs, database_embs)

                        rank = torch.argsort((-sim_mat).squeeze())

                        num_evaluated += 1

                        flag = 0
                        for j in range(len(rank)):
                            # if rank[j] == 0:
                            #     flag = 1
                            #     continue
                            if labels[1:][rank[j]] in true_neighbors:
                                if j == 0:
                                    similarity = sim_mat[rank[j]]
                                    top1_similarity_score.append(similarity)
                                recall[j - flag] += 1
                                break

                        # if len(list(set(indices[0][0:threshold]).intersection(set(true_neighbors)))) > 0:
                        #     one_percent_retrieved += 1

                    # one_percent_recall = (one_percent_retrieved/float(num_evaluated))*100
                    recall = (np.cumsum(recall) / float(num_evaluated)) * 100
                    max_ = max(max_, recall[0])
                    print(" ")
                    print("time: ", np.mean(timings))
                    print("recall\n", recall)
                    # print(t_loss / num_evaluated)
                    print("max:", max_)
                    # print(gt_iou.view(-1,)[:len(pos_mask[0])])

        # tbar.set_postfix({'train loss': loss / cnt})
