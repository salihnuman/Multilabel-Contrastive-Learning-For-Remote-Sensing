"""
Author: Yonglong Tian (yonglong@mit.edu)
Date: May 07, 2020
"""
import torch
from torch import nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""

    def __init__(self, temperature=0.07, contrast_mode="all", base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = torch.device("cuda") if features.is_cuda else torch.device("cpu")

        if len(features.shape) < 3:
            raise ValueError(
                "`features` needs to be [bsz, n_views, ...],"
                "at least 3 dimensions are required"
            )
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError("Cannot define both `labels` and `mask`")
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)

            if labels.shape[0] != batch_size:
                raise ValueError("Num of labels does not match num of features")
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        # 2
        contrast_count = features.shape[1]
        
        # 32,128
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        
        if self.contrast_mode == "one":
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == "all":   # default
            anchor_feature = contrast_feature   # 32,128
            anchor_count = contrast_count       # 2
        else:
            raise ValueError("Unknown mode: {}".format(self.contrast_mode))

        # compute logits
        # 32,32
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T), self.temperature
        )
        
        # for numerical stability
        # 32,1
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        # 32,32
        mask = mask.repeat(anchor_count, contrast_count)

        # mask-out self-contrast cases
        # 32,32
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0,
        )
        
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

class MulSupCosineLossCustom(nn.Module):
    """
    An attempt at multilabel supervised loss
    """

    def __init__(self, device, temperature=0.07, contrast_mode="all", base_temperature=0.07):
        super(MulSupCosineLossCustom, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.mask = None
        self.device = device
        self.cos = nn.CosineSimilarity(dim=0)

    def forward(self, features, labels):
        """Compute loss for model. 

        features: hidden vector of shape [bsz, n_views, ...].
        labels: ground truth of shape [bsz].
        mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
            has the same class as sample i. Can be asymmetric.
        
        """

        if len(features.shape) < 3:
            raise ValueError(
                "`features` needs to be [bsz, n_views, ...],"
                "at least 3 dimensions are required"
            )
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        # expected shape of features: 16,2,128

        batch_size = features.shape[0]
        
        # now we need to create mask
        # as a matrix of size 16x16
        # where True is assigned only to cells i,j
        # such that samples i and i are considered 
        # similar/positive/of the same class

        # labels 16,17
    
        self.mask = torch.eye(batch_size,dtype=bool)

        # ALL: meaning that ALL labels must match
        """ for i in range(batch_size):
                for j in range(batch_size):
                    self.mask[i][j] =  torch.equal(labels[i],labels[j]) """

        # ANY: meaning that they are considered similar, even if a single label matches
        """ for i in range(batch_size):
            for j in range(i+1):
                intersection = torch.logical_and(labels[i],labels[j])
                self.mask[i][j] = (~torch.all(intersection == False)).item()
                self.mask[j][i] = self.mask[i][j] """
        
        # COSINE
        self.mask = torch.eye(batch_size,dtype=float)
        for i in range(batch_size):
            for j in range(i+1):
                cosineSim = self.cos(labels[i],labels[j])
                self.mask[i][j] = cosineSim
                self.mask[j][i] = self.mask[i][j]

        self.mask = self.mask.float().to(self.device)

        # the rest of the code is as is.
        # 2
        contrast_count = features.shape[1]
        
        # 32,128
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        
        if self.contrast_mode == "one":
            # 16, 128
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == "all":
            # 32, 128
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError("Unknown mode: {}".format(self.contrast_mode))

        # compute logits
        # 32,32: dot products between all pairs of feature vectors
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T), self.temperature
        )

        # for numerical stability
        # 32,1
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        self.mask = self.mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(self.mask),1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(self.device),0)
        
        # masks out the diagonal of the mask
        self.mask = self.mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask

        # logits = torch.log(exp_logits); nicely done.
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (self.mask * log_prob).sum(1) / self.mask.sum(1)

        # loss
        # 32
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos

        # 2,16, why the view?
        loss = loss.view(anchor_count, batch_size).mean()
        
        return loss
    
class MulSupCosineLossCustomOneCrop(nn.Module):
    """
    An attempt at multilabel supervised loss
    """

    def __init__(self, device, temperature=0.07, contrast_mode="all", base_temperature=0.07):
        super(MulSupCosineLossCustomOneCrop, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.mask = None
        self.device = device
        self.cos = nn.CosineSimilarity(dim=0)

    def forward(self, features, labels):
        """Compute loss for model. 

        features: hidden vector of shape [bsz, n_views, ...].
        labels: ground truth of shape [bsz].
        mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
            has the same class as sample i. Can be asymmetric.
        
        """
        """
        if len(features.shape) < 3:
            raise ValueError(
                "`features` needs to be [bsz, n_views, ...],"
                "at least 3 dimensions are required"
            )
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)
        """
        # expected shape of features: 16,128

        batch_size = features.shape[0]
        
        # now we need to create mask
        # as a matrix of size 16x16
        # where True is assigned only to cells i,j
        # such that samples i and i are considered 
        # similar/positive/of the same class

        # labels 16,17
    
        self.mask = torch.eye(batch_size,dtype=bool)

        # ALL: meaning that ALL labels must match
        # for i in range(batch_size):
        #     for j in range(batch_size):
        #         self.mask[i][j] =  torch.equal(labels[i],labels[j])

        # ANY: meaning that they are considered similar, even if a single label matches
        # for i in range(batch_size):
        #     for j in range(i+1):
        #         intersection = torch.logical_and(labels[i],labels[j])
        #         self.mask[i][j] = (~torch.all(intersection == False)).item()
        #         self.mask[j][i] = self.mask[i][j]
        
        # COSINE
        # self.mask = torch.eye(batch_size,dtype=float)
        # for i in range(batch_size):
        #     for j in range(i+1):
        #         cosineSim = self.cos(labels[i],labels[j])
        #         self.mask[i][j] = cosineSim
        #         self.mask[j][i] = self.mask[i][j]
        
        self.mask = F.cosine_similarity(
            labels.unsqueeze(1), labels.unsqueeze(0), dim=2
        )  # Shape: [batch_size, batch_size]

        # JACCARD
        # self.mask = torch.eye(batch_size,dtype=float)
        # for i in range(batch_size):
        #     for j in range(i+1):
        #         intersection = torch.logical_and(labels[i],labels[j])
        #         union = torch.logical_or(labels[i],labels[j])
        #         jaccardSim = torch.sum(intersection) / torch.sum(union)
        #         self.mask[i][j] = jaccardSim
        #         self.mask[j][i] = self.mask[i][j]

        # Hamming
        # self.mask = torch.eye(batch_size,dtype=float)
        # for i in range(batch_size):
        #     for j in range(i+1):
        #         hammingDist = torch.sum(torch.logical_xor(labels[i],labels[j]))
        #         hammingSim = 1 - hammingDist / labels.shape[1]
        #         self.mask[i][j] = hammingSim
        #         self.mask[j][i] = self.mask[i][j]

        self.mask = self.mask.float().to(self.device)

        
        # the rest of the code is as is.
        # 2
        
        contrast_count = features.shape[1]
        
        # 16,128
        contrast_feature = features
        anchor_feature = features
        """
        if self.contrast_mode == "one":
            # 16, 128
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == "all":
            # 32, 128
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError("Unknown mode: {}".format(self.contrast_mode))
        """
        # compute logits
        # 32,32: dot products between all pairs of feature vectors
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T), self.temperature
        )

        # for numerical stability
        # 32,1
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        #self.mask = self.mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(self.mask),1,
            torch.arange(batch_size).view(-1, 1).to(self.device),0)
        
        # masks out the diagonal of the mask
        self.mask = self.mask * logits_mask
        
        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask

        # logits = torch.log(exp_logits); nicely done.
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        
        # compute mean of log-likelihood over positive
        e = 1e-7
        mean_log_prob_pos = (self.mask * log_prob).sum(1) / (self.mask.sum(1) + e)
        #print(mean_log_prob_pos)
        # loss
        # 32
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        
        # 2,16, why the view?
        loss = loss.view(1, batch_size).mean()
        
        return loss

class MulSupConLossCustom(nn.Module):
    def __init__(self, device, temperature=0.07, contrast_mode="all", base_temperature=0.07):
        super(MulSupConLossCustom, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.mask = None
        self.device = device
        self.cos = nn.CosineSimilarity(dim=0)
        self.label_loss = SupConLoss(temperature=temperature, contrast_mode=contrast_mode, base_temperature=base_temperature)
    
    def forward(self, features, labels):
        loss = 0
        num_labels = labels.shape[1]
        for i in range(num_labels):
            # Calculate the SupCon loss for each label separately
            labels_i = labels[:, i]
            loss += self.label_loss(features, labels_i)

        # Paper divides the total loss to number of positive labels
        loss /= labels.sum()
        return loss
    
class WeightedMulSupConLossCustom(nn.Module):
    def __init__(self, device, weight_list, temperature=0.07, contrast_mode="all", base_temperature=0.07):
        super(WeightedMulSupConLossCustom, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.mask = None
        self.device = device
        self.cos = nn.CosineSimilarity(dim=0)
        self.label_loss = SupConLoss(temperature=temperature, contrast_mode=contrast_mode, base_temperature=base_temperature)
        self.weight_list = weight_list
    
    def forward(self, features, labels):
        loss = 0
        num_labels = labels.shape[1]
        for i in range(num_labels):
            # Calculate the SupCon loss for each label separately
            labels_i = labels[:, i]
            loss += self.weight_list[i] * self.label_loss(features, labels_i)

        # Paper divides the total loss to number of positive labels
        loss /= labels.sum()
        return loss
    

class MulSupCosineLoss(torch.nn.Module):
    def __init__(self, t, device="cuda:0") -> None:
        super().__init__()
        self.t = t
        self.device = device

    def forward(self, embeddings, labels):
        batch_size = embeddings.size(0)

        # Compute cosine similarities for all pairs in the batch
        cosine_sim_matrix = F.cosine_similarity(
            embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=2
        )  # Shape: [batch_size, batch_size]
        
        # Exponentiate and scale cosine similarities
        exp_cosine_sim_matrix = torch.exp(cosine_sim_matrix / self.t)  # Shape: [batch_size, batch_size]
        
        # Mask out self-similarities (diagonal)
        mask =  torch.scatter(
            torch.ones(batch_size, batch_size).to(self.device),1,
            torch.arange(batch_size).view(-1, 1).to(self.device),0)
        
        exp_cosine_sim_matrix = exp_cosine_sim_matrix * mask
        
        # Denominator for all samples
        denominator = exp_cosine_sim_matrix.sum(dim=1, keepdim=True)  # Shape: [batch_size, 1]
        
        # Compute weights based on label similarities
        label_sim_matrix = F.cosine_similarity(
            labels.unsqueeze(1), labels.unsqueeze(0), dim=2
        )  # Shape: [batch_size, batch_size]

        # Mask out self-similarities in label similarity matrix
        label_sim_matrix = label_sim_matrix * mask
        

        # Numerator for each pair
        numerator = exp_cosine_sim_matrix  # Shape: [batch_size, batch_size]
        numerator += 1e-12  # Avoid zero for log

        # Weighted log-probability loss
        weighted_log_prob = -torch.log(numerator / denominator) * label_sim_matrix  # Shape: [batch_size, batch_size]
        
        # Normalize each row's loss by the total weight
        total_weights = label_sim_matrix.sum(dim=1, keepdim=True)  # Shape: [batch_size, 1]
        total_weights = total_weights + 1e-12  # Avoid division by zero
        normalized_loss = weighted_log_prob.sum(dim=1) / total_weights.squeeze()

        # Average over all samples
        loss = normalized_loss.mean()

        return loss
    

class WeightedMulSupCosineLossCustom(torch.nn.Module):
    def __init__(self, t=0.1, co_oc_mat=None, device="cuda:0", base_temperature=0.07):
        super(WeightedMulSupCosineLossCustom, self).__init__()
        self.temperature = t
        self.base_temperature = base_temperature
        self.device = device
        self.co_oc_mat = torch.tensor(co_oc_mat, device=device, dtype=torch.float32) if co_oc_mat is not None else None
    
    def forward(self, embeddings, labels):

        # Compute fuzzy labels based on co-occurrence matrix
        labels = labels.float()
        fuzzy_labels = torch.matmul(labels, self.co_oc_mat)  # Shape: [batch_size, num_labels]

        # Go with the fuzzy label matrix to calculate cosine loss
        batch_size = embeddings.shape[0]
    
        self.mask = torch.eye(batch_size,dtype=bool)
        
        self.mask = F.cosine_similarity(
            fuzzy_labels.unsqueeze(1), fuzzy_labels.unsqueeze(0), dim=2
        )  # Shape: [batch_size, batch_size]

        self.mask = self.mask.float().to(self.device)

        
        # the rest of the code is as is.
        # 2
        
        contrast_count = embeddings.shape[1]
        
        # 16,128
        contrast_feature = embeddings
        anchor_feature = embeddings
        
        # compute logits
        # 32,32: dot products between all pairs of feature vectors
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T), self.temperature
        )

        # for numerical stability
        # 32,1
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        #self.mask = self.mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(self.mask),1,
            torch.arange(batch_size).view(-1, 1).to(self.device),0)
        
        # masks out the diagonal of the mask
        self.mask = self.mask * logits_mask
        
        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask

        # logits = torch.log(exp_logits); nicely done.
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        
        # compute mean of log-likelihood over positive
        e = 1e-7
        mean_log_prob_pos = (self.mask * log_prob).sum(1) / (self.mask.sum(1) + e)
        #print(mean_log_prob_pos)
        # loss
        # 32
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        
        # 2,16, why the view?
        loss = loss.view(1, batch_size).mean()
        
        return loss
