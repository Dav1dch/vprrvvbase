import torch


def sigmoid(tensor, temp=1.0):
    """temperature controlled sigmoid
    takes as input a torch tensor (tensor) and passes it through a sigmoid, controlled by temperature: temp
    """
    exponent = -tensor / temp
    # clamp the input tensor for stability
    exponent = torch.clamp(exponent, min=-50, max=50)
    y = 1.0 / (1.0 + torch.exp(exponent))
    return y


class ERFA(torch.nn.Module):
    """PyTorch implementation of the Smooth-AP loss.
    implementation of the Smooth-AP loss. Takes as input the mini-batch of CNN-produced feature embeddings and returns
    the value of the Smooth-AP loss. The mini-batch must be formed of a defined number of classes. Each class must
    have the same number of instances represented in the mini-batch and must be ordered sequentially by class.
    e.g. the labels for a mini-batch with batch size 9, and 3 represented classes (A,B,C) must look like:
        labels = ( A, A, A, B, B, B, C, C, C)
    (the order of the classes however does not matter)
    For each instance in the mini-batch, the loss computes the Smooth-AP when it is used as the query and the rest of the
    mini-batch is used as the retrieval set. The positive set is formed of the other instances in the batch from the
    same class. The loss returns the average Smooth-AP across all instances in the mini-batch.
    Args:
        anneal : float
            the temperature of the sigmoid that is used to smooth the ranking function. A low value of the temperature
            results in a steep sigmoid, that tightly approximates the heaviside step function in the ranking function.
        batch_size : int
            the batch size being used during training.
        num_id : int
            the number of different classes that are represented in the batch.
        feat_dims : int
            the dimension of the input feature embeddings
    Shape:
        - Input (preds): (batch_size, feat_dims) (must be a cuda torch float tensor)
        - Output: scalar
    Examples::
        >>> loss = SmoothAP(0.01, 60, 6, 256)
        >>> input = torch.randn(60, 256, requires_grad=True).cuda()
        >>> output = loss(input)
        >>> output.backward()
    """

    def __init__(self):
        """
        Parameters
        ----------

        """
        super(ERFA, self).__init__()

    def ndcg(self, golden, current, n=-1):
        # 第一个log2(i + 1) = log2(1 + 1) = 1
        log2_table = torch.log2(torch.arange(2, 102))
        log2_table = log2_table.to("cuda")

        def dcg_at_n(rel, n):
            rel = rel[:n]  # 转numpy float数组，并取前n个值
            # dcg由cg加权求和而来
            #  此处分子对rel_i进行了指数放大
            #  权重为log2_i，使用log2_table预先计算再截取的形式进行加速
            dcg = torch.sum(
                torch.divide(torch.pow(2, rel) - 1, log2_table[: rel.shape[0]])
            )
            return dcg

        # 共len(current)个搜索结果用来评估搜索引擎
        # 最后给出搜索引擎的ndcg值，是各个搜索结果的ndcg的平均
        ndcgs = []
        for i in range(len(current)):
            # 如果规定了n，就计算ndcg@n；如果没有，就计算ndcg@len(current[i])
            k = len(current[i]) if n == -1 else n
            idcg = dcg_at_n(
                torch.sort(golden[i], descending=True)[0], n=k
            )  # 计算idcg@k
            dcg = dcg_at_n(current[i], n=k)  # 计算dcg@k
            tmp_ndcg = 0.0 if idcg == 0.0 else dcg / idcg  # 计算当前搜索结果的ndcg@k
            ndcgs.append(tmp_ndcg)
        ndcgs = torch.tensor(ndcgs, requires_grad=True)
        # 计算所有搜索结果的ndcg的平均值
        return (
            torch.zeros((1,), requires_grad=True)
            if len(ndcgs) == 0
            else torch.sum(ndcgs) / (len(ndcgs))
        )

    def forward(self, sim_all, pos_mask_, gt_iou):
        """Forward pass for all input predictions: preds - (batch_size x feat_dims)"""
        """_summary_

        Example: pred: [1, 0.9, 0.7, 0.6 0.3, 0.2]
        gt: [1, 1, 0, 1, 0, 0] smoothap = 0.833  forward = 0.833
        gt: [1, 1, 1, 1, 0, 0] smoothap = 1      forward = 1
        gt: [1, 1, 0, 1, 0, 1] smoothap = 0.7555 forward = 0.755

        Returns:
            _type_: _description_
        """
        # sim_mask = sim_all[1:]
        sim_mask = sim_all
        pos_mask = pos_mask_[:, 1:].to("cuda")
        neg_mask = (~pos_mask).to(torch.float).to("cuda")
        # rel = most_pos[:, 1:].to(torch.float).to('cuda')
        rel = pos_mask_[:, 1:].to(torch.float).to("cuda")

        sort_ind = torch.argsort(-sim_mask)
        neg_mask[0] = neg_mask[0][sort_ind]
        rel[0] = rel[0][sort_ind]
        gt_iou = gt_iou.cuda()[sort_ind]
        ndcg = self.ndcg(gt_iou.unsqueeze(0), gt_iou.unsqueeze(0))
        neg_ndcg = self.ndcg(neg_mask, neg_mask)
        if torch.sum(pos_mask) == 0:
            return torch.tensor(0.0001, requires_grad=True).cuda()

        d = sim_mask.squeeze().unsqueeze(0)
        d_repeat = d.repeat(len(sim_mask), 1)
        D = d_repeat - d_repeat.T
        D = sigmoid(D, 0.01)
        D_ = D * (1 - torch.eye(len(sim_mask))).to("cuda")
        D_pos = D_ * pos_mask

        R = 1 + torch.sum(D_, 1)
        R_pos = (1 + torch.sum(D_pos, 1) * ndcg) * pos_mask
        R_neg = R - R_pos
        R_ndcg = R_neg * neg_ndcg + R_pos
        R = R_neg + R_pos

        ap = torch.zeros(1, requires_grad=True).cuda()
        ap_ndcg = (1 / torch.sum(pos_mask)) * torch.sum(R_pos / R_ndcg)
        # ap_nondcg = (1 / torch.sum(pos_mask)) * torch.sum(R_pos / R)

        ap_ndcg = ap + ap_ndcg
        # print("ndcg", ap_ndcg * ndcg)

        return ap_ndcg


class SmoothAP(torch.nn.Module):
    """PyTorch implementation of the Smooth-AP loss.
    implementation of the Smooth-AP loss. Takes as input the mini-batch of CNN-produced feature embeddings and returns
    the value of the Smooth-AP loss. The mini-batch must be formed of a defined number of classes. Each class must
    have the same number of instances represented in the mini-batch and must be ordered sequentially by class.
    e.g. the labels for a mini-batch with batch size 9, and 3 represented classes (A,B,C) must look like:
        labels = ( A, A, A, B, B, B, C, C, C)
    (the order of the classes however does not matter)
    For each instance in the mini-batch, the loss computes the Smooth-AP when it is used as the query and the rest of the
    mini-batch is used as the retrieval set. The positive set is formed of the other instances in the batch from the
    same class. The loss returns the average Smooth-AP across all instances in the mini-batch.
    Args:
        anneal : float
            the temperature of the sigmoid that is used to smooth the ranking function. A low value of the temperature
            results in a steep sigmoid, that tightly approximates the heaviside step function in the ranking function.
        batch_size : int
            the batch size being used during training.
        num_id : int
            the number of different classes that are represented in the batch.
        feat_dims : int
            the dimension of the input feature embeddings
    Shape:
        - Input (preds): (batch_size, feat_dims) (must be a cuda torch float tensor)
        - Output: scalar
    Examples::
        >>> loss = SmoothAP(0.01, 60, 6, 256)
        >>> input = torch.randn(60, 256, requires_grad=True).cuda()
        >>> output = loss(input)
        >>> output.backward()
    """

    def __init__(self):
        """
        Parameters
        ----------

        """
        super(SmoothAP, self).__init__()

    def ndcg(self, golden, current, n=-1):
        # 第一个log2(i + 1) = log2(1 + 1) = 1
        log2_table = torch.log2(torch.arange(2, 102))
        log2_table = log2_table.to("cuda")

        def dcg_at_n(rel, n):
            rel = rel[:n]  # 转numpy float数组，并取前n个值
            # dcg由cg加权求和而来
            #  此处分子对rel_i进行了指数放大
            #  权重为log2_i，使用log2_table预先计算再截取的形式进行加速
            dcg = torch.sum(
                torch.divide(torch.pow(2, rel) - 1, log2_table[: rel.shape[0]])
            )
            return dcg

        # 共len(current)个搜索结果用来评估搜索引擎
        # 最后给出搜索引擎的ndcg值，是各个搜索结果的ndcg的平均
        ndcgs = []
        for i in range(len(current)):
            # 如果规定了n，就计算ndcg@n；如果没有，就计算ndcg@len(current[i])
            k = len(current[i]) if n == -1 else n
            idcg = dcg_at_n(
                torch.sort(golden[i], descending=True)[0], n=k
            )  # 计算idcg@k
            dcg = dcg_at_n(current[i], n=k)  # 计算dcg@k
            tmp_ndcg = 0.0 if idcg == 0.0 else dcg / idcg  # 计算当前搜索结果的ndcg@k
            ndcgs.append(tmp_ndcg)
        ndcgs = torch.tensor(ndcgs, requires_grad=True)
        # 计算所有搜索结果的ndcg的平均值
        return (
            torch.zeros((1,), requires_grad=True)
            if len(ndcgs) == 0
            else torch.sum(ndcgs) / (len(ndcgs))
        )

    def forward(self, sim_all, pos_mask_, gt_iou):
        """Forward pass for all input predictions: preds - (batch_size x feat_dims)"""
        """_summary_

        Example: pred: [1, 0.9, 0.7, 0.6 0.3, 0.2]
        gt: [1, 1, 0, 1, 0, 0] smoothap = 0.833  forward = 0.833
        gt: [1, 1, 1, 1, 0, 0] smoothap = 1      forward = 1
        gt: [1, 1, 0, 1, 0, 1] smoothap = 0.7555 forward = 0.755

        Returns:
            _type_: _description_
        """
        # sim_mask = sim_all[1:]
        sim_mask = sim_all
        pos_mask = pos_mask_[:, 1:].to("cuda")
        neg_mask = (~pos_mask).to(torch.float).to("cuda")
        # rel = most_pos[:, 1:].to(torch.float).to('cuda')
        rel = pos_mask_[:, 1:].to(torch.float).to("cuda")

        sort_ind = torch.argsort(-pos_mask.type(torch.float)[0])
        # sort_ind = torch.argsort(-sim_mask)
        neg_mask[0] = neg_mask[0][sort_ind]
        # ndcg_neg = self.ndcg(neg_mask, neg_mask)
        rel[0] = rel[0][sort_ind]
        # ndcg = self.ndcg(rel, rel)
        if torch.sum(pos_mask) == 0:
            return torch.tensor(0.0001, requires_grad=True).cuda()

        d = sim_mask.squeeze().unsqueeze(0)
        d_repeat = d.repeat(len(sim_mask), 1)
        D = d_repeat - d_repeat.T
        D = sigmoid(D, 0.01)
        D_ = D * (1 - torch.eye(len(sim_mask))).to("cuda")
        D_pos = D_ * pos_mask

        R = 1 + torch.sum(D_, 1)
        R_pos = (1 + torch.sum(D_pos, 1)) * pos_mask
        R_neg = R - R_pos
        # R_ndcg = R_neg * neg_ndcg + R_pos
        R = R_neg + R_pos

        ap = torch.zeros(1, requires_grad=True).cuda()
        # ap_ndcg = (1 / torch.sum(pos_mask)) * torch.sum(R_pos / R_ndcg)
        ap_nondcg = (1 / torch.sum(pos_mask)) * torch.sum(R_pos / R)

        ap_nondcg = ap + ap_nondcg
        # print("ndcg", ap_ndcg * ndcg)

        return ap_nondcg
