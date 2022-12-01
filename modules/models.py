import torch
from torch.nn.functional import normalize


# Generalized Model for our Contrastive Clustering Method
class Model(torch.nn.Module):
    
    def __init__ (self, backbone, latent_dim, num_classes) : 
        
        super(Model, self).__init__()
        self.backbone = backbone
        self.latent_dim = latent_dim
        self.num_clusters = num_classes
        self.relu = torch.nn.ReLU
        
        # Network for Instance-Level Contrastive Head
        self.instance_projector = torch.nn.Sequential(
            torch.nn.Linear(self.backbone.rep_dim, self.backbone.rep_dim),
            self.relu(),
            torch.nn.Linear(self.backbone.rep_dim, self.latent_dim),
        )
        
        
        # Network for Cluster-Level Contrastive Head
        self.cluster_projector = torch.nn.Sequential(
            torch.nn.Linear(self.backbone.rep_dim, self.backbone.rep_dim),
            self.relu(),
            torch.nn.Linear(self.backbone.rep_dim, self.num_classes),
            torch.nn.Softmax(dim=1)
        )
        
        
    def forward(self, X_1, X_2) : 
        # Encoding the X_1 and X_2 using our backbone network
        H_1 = self.backbone(X_1)
        H_2 = self.backbone(X_2)
        
        # Getting Instance level projections
        Z_1 = normalize(self.instance_projector(H_1), dim=1)
        Z_2 = normalize(self.instance_projector(H_2), dim=1)
        
        # Getting Cluster Level projections
        C_1 = self.cluster_projector(H_1)
        C_2 = self.cluster_projector(H_2)
        
        return Z_1, Z_2, C_1, C_2
    
    
    def forward_cluster(self, X) : 
        # For assigning clusters 
        H = self.backbone(X)
        C = self.cluster_projector(H)
        return torch.argmax(C, dim=1)