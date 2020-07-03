import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np


def get_data_loaders(root_dir, batch_size, augmented, 
                    random_seed = True,
                    random_seed_val = 0,
                    num_workers = 1,
                    dev_ratio = 0.1,
                    shuffle = True ):
    
    mean=[0.4914, 0.4822, 0.4465]
    std=[0.2023, 0.1994, 0.2010]
    norm = transforms.Normalize(mean, std)


    ## getting the transforms
    if augmented:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding = 4, pad_if_needed = True, padding_mode = 'reflect'),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            norm
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            norm
        ])

    dev_transform = transforms.Compose([
        transforms.ToTensor(),
        norm
    ])

    
    ## extracting training set
    train_set = torchvision.datasets.CIFAR10(
        root = root_dir,
        train = True,
        download = True,
        transform = train_transform
    )

    ## extracting dev set
    dev_set = torchvision.datasets.CIFAR10(
        root = root_dir,
        train = True,
        download = True,
        transform = dev_transform
    )


    ## getting the samplers
    train_length = len(train_set)
    ids = list(range(train_length))
    split_id = int(np.floor(train_length*dev_ratio))

    if shuffle and random_seed:
        np.random.seed(random_seed_val)
        np.random.shuffle(ids)

    train_ids, dev_ids = ids[split_id:], ids[:split_id]
    train_sampler = SubsetRandomSampler(train_ids)
    dev_sampler = SubsetRandomSampler(dev_ids)

    
    ## getting the loaders
    train_loader = DataLoader(train_set, batch_size = batch_size, num_workers = num_workers, sampler = train_sampler)
    dev_loader = DataLoader(dev_set, batch_size = batch_size, num_workers = num_workers, sampler = dev_sampler)

    return (train_loader, dev_loader)



def get_test_loader(root_dir, batch_size,
                    shuffle = True,
                    num_workers = 1,
                    augmented = True):    

    mean_test = [0.485, 0.456, 0.406]
    std_test = [0.229, 0.224, 0.225]    
    norm_test = transforms.Normalize(mean_test, std_test) 


    ## obtaining test transform
    if augmented:
        test_transform = transforms.Compose([
            transforms.RandomCrop(32, padding = 4, pad_if_needed = True, padding_mode = 'reflect'),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            norm_test
        ])
    else:
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            norm_test
        ])

    ## extracting test set
    test_set = torchvision.datasets.CIFAR10(
        root = root_dir,
        train = False,
        download = True,
        transform = test_transform
    )

    test_loader = DataLoader(test_set, batch_size = batch_size, num_workers = num_workers)

    return test_loader, test_set.classes


