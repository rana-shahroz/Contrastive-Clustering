import os
import torch
import json
import numpy as np
from utils import read_config, data_loader, trainer, trainer2
from modules import resnet, models, losses

if __name__ == '__main__' : 
    # Reading Config File
    args = read_config.get_args(filename = 'Configs/config.yaml')
    
    # Setting up seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    
    # Getting data loaders
    train_data = data_loader.get_data_train(args)
    test_data = data_loader.get_data_test(args)

    # Initializing Models and optimizer
    res = resnet.get_resnet(args.resnet)
    tmp = models.Model(res, 128, 5)
    tmp = torch.nn.DataParallel(tmp)
    tmp = tmp.to(args.device)
    weights_mnist_vanilla = os.path.join("save/VanillaK=5/iter1", f"checkpoint_{200}.tar")
    chkpt = torch.load(weights_mnist_vanilla)
    tmp.load_state_dict(chkpt['net']) 
    for name, param in tmp.named_parameters():
        if "module.backbone" in name : 
            param.requires_grad = False
    acc_k = []
    for k in range(3,12) : 
        print(f"************K = {k}")
        args.num_classes = k
        model = models.Model(tmp.module.backbone, args.feature_dim, args.num_classes)
        model = torch.nn.DataParallel(model)
        model = model.to(args.device)
        
        for name, param in model.named_parameters() : 
            if "module.backbone" in name : 
                param.requires_grad = False
        optimizer = torch.optim.Adam(model.parameters(), lr = args.learning_rate, weight_decay = 0)
        
        # For continuing where we left off
        if args.continue_before : 
            weights = os.path.join(args.model_path, f"checkpoint_{args.start_epoch}.tar")
            chkpt = torch.load(weights)
            model.load_state_dict(chkpt['net'])
            optimizer.load_state_dict(chkpt['optimizer'])
            args.start_epoch = chkpt['epoch'] + 1

        instance_loss = losses.InstanceLoss(args.batch_size, args.instance_temperature, args.device).to(args.device)
        cluster_loss = losses.ClusterLoss(args.num_classes, args.cluster_temperature, args.device, similarity_metric = args.metric).to(args.device)
        
        # Training Loop
        acc, nmi, ari = trainer.train(args, train_data, test_data, optimizer, model, 
                                instance_loss, cluster_loss, args.device)
        best_acc = acc['20']
        acc_k.append(best_acc)
        print(f"K = {k}\tacc={best_acc}")
        
        
    # Saving the acc, nmi, ari in json files.
    acc_file = os.path.join(args.model_path, 'acc.json') 
    nmi_file = os.path.join(args.model_path, 'nmi.json') 
    ari_file = os.path.join(args.model_path, 'ari.json') 
    acc_k_file = os.path.join(args.model_path, 'acc_k.json')
    
    json_acc = json.dumps(acc)
    json_nmi = json.dumps(nmi)
    json_ari = json.dumps(ari)
    json_acc_k = json.dumps(acc_k)
    
    with open(acc_file, "w") as outfile:
        outfile.write(json_acc)
        
    with open(nmi_file, "w") as outfile:
        outfile.write(json_nmi)
    
    with open(ari_file, "w") as outfile:
        outfile.write(json_ari)
        
    with open(acc_k_file, "w") as outfile:
        outfile.write(json_acc_k)