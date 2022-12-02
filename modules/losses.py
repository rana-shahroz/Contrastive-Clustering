import torch.nn as nn
import torch
import math
from modules.distances import get_similarity_metric


# Implementing the Loss for the Instance Contrastive Head 
# Explanation and Background : 
#   Normally contrastive learning repels the pairs from different class and 
#   attracks pairs from same class. We do not have labels in this case.
#   So we work with pseudo labels and augmentations. Feature matrix is created but
#   their is a lot of information loss as we are pseudo labelling and augmenting. 
#   So we use a mlp for h to z transformation on the feature matrix and apply this 
#   contrastive loss their.

# Algorithm : 
#   1. Given N samlpes we perform 2 kind of augmentations on each to get 2N data samples.
#   2. For one sample x_i^a we now choose from 2N-1 pairs the pair that is made from its
#       augmentation x_i^b as positive pair. Rest are masked as negative for this specific
#       sample x_i. [a and b are the two data augmentations.] -> This is the masked 
#       feature matrix. 
#   3. This feature matrix is passed over the MLP in the model class [self.instance_projector].
#   4. -> We are HERE : Calculating the loss for the augmented dataset as follows : 
#           l_i^a = -log [exp (similarity_metric(z_i^a, z_i^b)/tao_I)/ (Sum for z_i^a, z_i^b)]
#       Technically just -log(P(sim(z_i^a, z_i^b))) = CrossEntropy lmao. 
#       tao_I : Temperature parameter.
class InstanceLoss(nn.Module):
    def __init__(self, batch_size, temperature, device):
        super(InstanceLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device

        self.mask = self.mask_correlated_samples(batch_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        mask = mask.bool()
        return mask

    def forward(self, z_i, z_j):
        N = 2 * self.batch_size
        z = torch.cat((z_i, z_j), dim=0)

        sim = torch.matmul(z, z.T) / self.temperature
        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N

        return loss



# Explanation and Background : 
#   If projecting a representation to the latent space that has is equal whose dimensions
#   is equal to the number of cluster, then the ith element of its feature represents the 
#   probability of belonging to the ith cluster. Hence our cluster contrastive head will
#   output Y^a \in R^N*M [N = batchsize, M = clusters] for first augmentation. As one 
#   point belongs to one cluster Y^a_n,m is one hot encoded. Do this for both Y^a and Y^b.

# Algorithm : 
#   1. Two layer MLP to project the feature matrix into an M-dimenstional space.
#   2. Make positive cluster pair while making 2M-2 to be negative pairs. Anything in the
#       cluster is a positive pair while outside is the negative pair.
#   3. We change the distances here : Cosine \ Pairwise.
#   4. To avoid trivial solutions we subtract entropy of cluster assignment probabilities.
class ClusterLoss(nn.Module):
    def __init__(self, class_num, temperature, device, similarity_metric):
        super(ClusterLoss, self).__init__()
        self.class_num = class_num
        self.temperature = temperature
        self.device = device

        self.mask = self.mask_correlated_clusters(class_num)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = get_similarity_metric(similarity_metric)

    def mask_correlated_clusters(self, class_num):
        N = 2 * class_num
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(class_num):
            mask[i, class_num + i] = 0
            mask[class_num + i, i] = 0
        mask = mask.bool()
        return mask

    def forward(self, c_i, c_j):
        p_i = c_i.sum(0).view(-1)
        p_i /= p_i.sum()
        ne_i = math.log(p_i.size(0)) + (p_i * torch.log(p_i)).sum()
        p_j = c_j.sum(0).view(-1)
        p_j /= p_j.sum()
        ne_j = math.log(p_j.size(0)) + (p_j * torch.log(p_j)).sum()
        ne_loss = ne_i + ne_j

        c_i = c_i.t()
        c_j = c_j.t()
        N = 2 * self.class_num
        c = torch.cat((c_i, c_j), dim=0)

        # print((self.similarity_f(c.unsqueeze(1), c.unsqueeze(0))).shape)
        sim = self.similarity_f(c.unsqueeze(1), c.unsqueeze(0)) / self.temperature
        sim_i_j = torch.diag(sim, self.class_num)
        sim_j_i = torch.diag(sim, -self.class_num)

        positive_clusters = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_clusters = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_clusters.device).long()
        logits = torch.cat((positive_clusters, negative_clusters), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N

        return loss + ne_loss
