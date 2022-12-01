import torch
import math
from distances import get_similarty_metric


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

class InstanceLoss(torch.nn.Module):
    
    def __init__(self, batch_size, temp, device) : 
        # Setting up args for computations
        super(InstanceLoss, self).__init__()
        self.batch_size = batch_size
        
        # Number of samples [after augmentation it should be 2 * batch]
        self.N = 2 * self.batch_size
        self.temp = temp
        self.device = device
        
        # Generating a correlation mask.
        self.mask = self._mask_correlated_samples()
        self.criterion = torch.nn.CrossEntropyLoss(reduction = "sum")
        
    def _mask_correlated_samples(self) : 
        mask = torch.ones((self.N,self.N))
        
        # Notice the diagonal of the mask is useless diag(mask) = (x_i,x_i) 
        # for all i, which is not needed.
        mask = mask.fill_diagonal_(0)
        
        for i in range(self.batch_size) :
            # Taking care of the augmentation. The augmentations are stacked in such a way
            # samples = [x_1^a, ..., x_N^a, x_1^b, ..., x_N^b] 
            mask[i, batch_size + i] = 0
            mask[batch + i, i] = 0
            
        return mask.bool()
    
    
    def forward(self, Z_i, Z_j) : 
        z = torch.cat((Z_i, Z_j), dim = 0)
        
        # Calculating the similarity using cosine distance 
        sim = torch.matmul(z, z.T) / self.temp
        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)
        
        # Calculating positive Samples and Negative Samples
        pos_samples = torch.cat((sim_i_j, sim_j_i), dim = 0).reshape(self.N, 1)
        neg_samples = sim[self.mask].reshape(N,-1)
        
        # Pseudo Samples
        labels = torch.zeros(self.N).to(self.device).long()
        output = torch.cat((pos_samples, neg_samples), dim = 1)
        loss = self.criterion(logits, labels)
        
        return loss /= self.N



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
class ClusterLoss(torch.nn.Module) : 
    
    def __init__(self, num_classes, temp, device, similarity_metric = 'cosine') :
        
        super(ClusterLoss, self).__init__()
        # Setting up args for computations
        self.num_classes = num_classes
        self.similarity_metric = get_similarty_metric(similarity_metric)
        self.temp = temp
        # Number of samples [after augmentation it should be 2 * clusters]
        self.N = 2 * num_classes
        self.device = device
        
        # Generating a correlation mask.
        self.mask = self._mask_correlated_samples()
        self.criterion = torch.nn.CrossEntropyLoss(reduction = 'sum')
        
    
    def _mask_correlated_samples(self) :
        mask = torch.ones((self.N,self.N))
        
        # Notice the diagonal of the mask is useless diag(mask) = (x_i,x_i) 
        # for all i, which is not needed.
        mask = mask.fill_diagonal_(0)
        
        for i in range(self.batch_size) :
            # Taking care of the augmentation. The augmentations are stacked in such a way
            # samples = [x_1^a, ..., x_N^a, x_1^b, ..., x_N^b] 
            mask[i, batch_size + i] = 0
            mask[batch + i, i] = 0
            
        return mask.bool()
    
    
    def forward(self, C_i, C_j):
        
        def calculate_entropyi(C):
            p = C.sum(0).view(-1)
            p /= p.sum()
            ne = math.log(p.size(0)) * (p * torch.log(p)).sum()
            return ne
        
        entropy_loss = calculate_entropyi(C_i) + calculate_entropyi(C_j)
        
        C_i = C_i.t()
        C_j = C_j.t()
        C = torch.cat((C_i, C_j), dim=0)
        
        # Calculating the similarity
        similarity = self.similarity_metric(c.unsqueeze(1), c.unsqueeze(0)) / self.temp
        sim_i_j = torch.diag(similarity, self.num_classes)
        sim_j_i = torch.diag(similarity, -self.num_classes)
        
        # Getting Positive and Negative Clusters
        pos_clusters = torch.cat((sim_i_j, sim_j_i), dim = 0).reshape(self.N, 1)
        neg_clusters = sim[self.mask].reshape(self.N,-1)
        
        # Getting labels, logits and calculating the loss
        labels = torch.zeros(self.N).to(self.device).long()
        logits = torch.cat((pos_clusters, neg_clusters), dim=1)
        loss = self.criterion(logits, labels)
        
        # Total loss is average_loss + entropy
        loss /= self.N
        
        return loss + entropy_loss