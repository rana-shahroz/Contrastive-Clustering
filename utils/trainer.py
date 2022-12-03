import torch
from tqdm import tqdm
from utils.evaluate_network import evaluate_model
from utils.save_model import save_current

def train(args, train_loader, test_loader, optimizer, network, instance_loss, cluster_loss, device) : 

    ACC = {}
    NMI = {}
    ARI = {}
    display_bar = tqdm(range(args.start_epoch, args.epochs), position=0, leave = True)
    for epoch in display_bar : 
        acc = 0
        nmi = 0
        ari = 0
        loss_e = 0
        steps = 0
        # Training one epoch
        for i, ((x_i, x_j), _) in enumerate(train_loader) : 
            steps += 1
            optimizer.zero_grad()
            x_i = x_i.to(device)
            x_j = x_j.to(device)
            z_i,z_j, c_i, c_j, a, _ = network(x_i, x_j)
            loss_i = instance_loss(z_i, z_j)
            loss_c = cluster_loss(c_i, c_j)
            loss = loss_i + loss_c
            loss.backward()
            optimizer.step()
            loss_e += loss.item()
            
            if steps % 100 == 0 : 
                display_bar.set_description(f"[Epochs: {epoch + 1}/{args.epochs - args.start_epoch}] "
                                f"Loss: {loss.item():.6f}" )
        
        if epoch % 5 == 0 or epoch == args.epochs-1 : 
            nmi, ari, acc = evaluate_model(args, network, test_loader, device)
            
            display_bar.set_description(f"[Epochs: {epoch + 1} / {args.epochs - args.start_epoch}]  "
                                f"Loss: {loss_e/steps:.6f}  ACC: {acc:.6f}  NMI: {nmi:.6f}  ARI: {ari:.6f}\t"
                                )
        
            ACC[epoch] = acc
            NMI[epoch] = nmi
            ARI[epoch] = ari
        
        
        if epoch % 10 == 0 : 
            save_current(args, network, optimizer, epoch)
            
    save_current(args, network, optimizer, args.epochs)
    
    return ACC, NMI, ARI