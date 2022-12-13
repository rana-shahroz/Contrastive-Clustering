import torch
from tqdm.notebook import tqdm
from utils.evaluate_network2 import evaluate_model
from utils.save_model import save_current
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from torch.nn.functional import normalize

def align_loss(x, y, alpha=2):
    return (x - y).norm(p=2, dim=1).pow(alpha).mean()


def uniform_loss(x, t=2):
    return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()


def train(train_loader, test_loader, optimizer, network, instance_loss, cluster_loss, device) : 

    ACC = {}
    NMI = {}
    ARI = {}
    display_bar = tqdm(range(200), position=0, leave = True)
    for epoch in display_bar : 
        acc = 0
        nmi = 0
        ari = 0
        loss_e = 0
        steps = 0
        network.train()
        # Training one epoch
        for i, ((x_i, x_j), _) in enumerate(train_loader) : 
            steps += 1
            optimizer.zero_grad()
            x_i = x_i.to(device)
            x_j = x_j.to(device)
            z_i,z_j, c_i, c_j, Ha, Hb = network(x_i, x_j)
            
            # Instance Level Loss
            loss_i = instance_loss(z_i, z_j)
            
            # Cluster Level Loss
            loss_c = cluster_loss(c_i, c_j)
            
            # Contrastive Loss
            loss_contrastive = loss_i + loss_c
            
            # Uniformity Loss 
            loss_uniformity = uniform_loss(Ha) + uniform_loss(Hb)
            
            # # Allign Loss 
            # loss_allign = align_loss(Ha, Hb)
            
            # Total Loss 

            loss = 0.95 * loss_contrastive  + 0.05 * loss_uniformity
            
            loss.backward()
            optimizer.step()
            loss_e += loss.item()
            
            if steps % 100 == 0 : 
                display_bar.set_description(f"[Epochs: {epoch + 1}/{200}] "
                                f"Loss: {loss.item():.6f}" )
        
#         if epoch % 5 == 0 or epoch == 199 : 
#             nmi, ari, acc = evaluate_model(network, test_loader, device)
            
#             display_bar.set_description(f"[Epochs: {epoch + 1} / {args.epochs - args.start_epoch}]  "
#                                 f"Loss: {loss_e/steps:.6f}  ACC: {acc:.6f}  NMI: {nmi:.6f}  ARI: {ari:.6f}\t"
#                                 )
        
#             ACC[epoch] = acc
#             NMI[epoch] = nmi
#             ARI[epoch] = ari
        
        if epoch % 10 == 0 : 
            network.eval()
            Has = []
            Hbs = []
            labels = np.zeros((273, 256))
            for i, ((x_i, x_j), l) in enumerate(train_loader) : 
                labels[i] = l.detach().cpu().numpy().flatten()
                z_i,z_j, c_i, c_j, Ha, Hb = network(x_i.to(device), x_j.to(device))
                Has.append(normalize(Ha).detach().cpu().numpy())
            x = torch.Tensor(Has)
            x = x.flatten(end_dim=1)
            x = np.asarray(x)
            x = x.T


            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')

            ax.scatter(x[0], x[1], x[2], c= labels, cmap ='tab10')
            ax.set_title(f"Epoch : {epoch}\tLoss={loss.item():.6f}")
            plt.show()
        
        # if epoch % 10 == 0 : 
        #     save_current(args, network, optimizer, epoch)
            
    # save_current(args, network, optimizer, args.epochs)
    
    return ACC, NMI, ARI