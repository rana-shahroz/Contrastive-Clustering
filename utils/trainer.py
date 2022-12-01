import torch
from tqdm import tqdm
from evaluate_network import evaluate_model
from save_model import save_current

def trainer(args, train_loader, test_loader, network, instance_loss, cluster_loss, device) : 
    
    optimizer = torch.optim.Adam(network.parameters(), lr = args.learning_rate, weight_decay = args.weight_decay)
    ACC = []
    NMI = []
    ARI = []
    display_bar = tqdm(range(args.start_epoch, args.epochs))
    for epoch in display_bar : 
        acc = 0
        nmi = 0
        ari = 0
        loss_e = 0
        
        # Training one epoch
        for i, ((xi, xj), _) in enumerate(train_loader) : 
            optimizer.zero_grad()
            x_i = x_i.to(device)
            x_j = x_j.to(device)
            z_i,z_j, c_i, c_j = model(x_i, x_j)
            loss_i = instance_loss(z_i, z_j)
            loss_c = cluster_loss(c_i, c_j)
            loss = loss_i + loss_c
            loss.backward()
            optimizer.step()
            loss_e += loss.item()
        
        nmi, ari, acc = evaluate_model(network, test_loader, device)
        
        display_bar.set_description(f"[Epochs : {epoch + 1} / {args.epochs - args.start_epoch}]\t"
                            f"Loss : {loss_e:.6f}\tACC : {acc:.6f}\tNMI =  : {nmi:.6f}\tARI =  : {ari:.6f}\t"
                            )
        ACC.append(acc)
        NMI.append(nmi)
        ARI.append(ari)
        
        if epoch % 10 == 0 : 
            save_current(args, network, optimizer, epoch)
            
    save_current(args, network, optimizer, args.epochs)