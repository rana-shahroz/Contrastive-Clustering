import torchvision
from utils import transform
from torch.utils import data


def get_data_train(args) : 
    
    if args.dataset == 'CIFAR10' : 
        train_dataset = torchvision.datasets.CIFAR10(
            root = args.dataset_dir,
            download = True,
            train = True, 
            transform = transform.Transforms(size = args.image_size, s = 0.5),
        )
        
        test_dataset = torchvision.datasets.CIFAR10(
            root = args.dataset_dir,
            download = True, 
            train = False, 
            transform = transform.Transforms(size = args.image_size, s = 0.5),
        )
        dataset = data.ConcatDataset([train_dataset, test_dataset])
    elif args.dataset == 'CIFAR100' : 
        train_dataset = torchvision.dataset.CIFAR100(
            root=args.dataset_dir,
            download=True,
            train=True,
            transform=transform.Transforms(size=args.image_size),
        )
    
        test_dataset = torchvision.dataset.CIFAR100(
            root=args.dataset_dir,
            download=True,
            train=False,
            transform=transform.Transforms(size=args.image_size),
        )
        dataset = data.ConcatDataset([train_dataset, test_dataset])
    else : 
        raise NotImplementedError
    
    data_loader = data.DataLoader(
        dataset, 
        batch_size = args.batch_size,
        shuffle = True,
        drop_last = True,
        num_workers = args.num_workers,
    )
    
    return data_loader



def get_data_test(args) : 
    
    if args.dataset == 'CIFAR10' : 
        train_dataset = torchvision.datasets.CIFAR10(
            root = args.dataset_dir,
            download = True,
            train = True, 
            transform = transform.Transforms(size = args.image_size, s = 0.5).test_transform,
        )
        
        test_dataset = torchvision.datasets.CIFAR10(
            root = args.dataset_dir,
            download = True, 
            train = False, 
            transform = transform.Transforms(size = args.image_size, s = 0.5).test_transform,
        )
        dataset = data.ConcatDataset([train_dataset, test_dataset])
    elif args.dataset == 'CIFAR100' : 
        train_dataset = torchvision.dataset.CIFAR100(
            root=args.dataset_dir,
            download=True,
            train=True,
            transform=transform.Transforms(size=args.image_size).test_transform,
        )
    
        test_dataset = torchvision.dataset.CIFAR100(
            root=args.dataset_dir,
            download=True,
            train=False,
            transform=transform.Transforms(size=args.image_size).test_transform,
        )
        dataset = data.ConcatDataset([train_dataset, test_dataset])
    else : 
        raise NotImplementedError
    
    data_loader = data.DataLoader(
        dataset, 
        batch_size = 500,
        shuffle = False,
        drop_last = False,
        num_workers = args.num_workers,
    )
    
    return data_loader