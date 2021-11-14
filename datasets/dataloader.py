from .transforms              import build_transform
from torchvision              import datasets
from torch.utils.data         import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

import numpy as np
import os

np.random.seed(11)

def make_train_loader(cfg):
    
    num_workers = cfg.DATA.NUM_WORKERS
    batch_size  = cfg.DATA.TRAIN_BATCH_SIZE
    valid_size  = cfg.DATA.VALIDATION_SIZE
    train_path  = cfg.PATH.TRAIN_SET
    
    transforms = build_transform(cfg)

    trainset = datasets.ImageFolder('C:/Users/user/Desktop/hw2/AIMango_sample/', transform=transforms)

    
    num_train = len(trainset)
    print('total_images: ',num_train)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)


    train_loader = DataLoader(trainset, batch_size=batch_size, num_workers=0, sampler=train_sampler)
    valid_loader = DataLoader(trainset, batch_size=batch_size, num_workers=0, sampler=valid_sampler)

    print('train_set done.')
    return train_loader, valid_loader

def make_test_loader(cfg):

    num_workers = cfg.DATA.NUM_WORKERS
    batch_size  = cfg.DATA.TEST_BATCH_SIZE
    test_path   = cfg.PATH.TEST_SET

    transforms = build_transform(cfg)

    testset = datasets.ImageFolder('C:/Users/user/Desktop/hw2/AIMango_sample/', transform=transforms)

    test_loader = DataLoader(testset, batch_size=batch_size, num_workers=num_workers)
    print('test_set done.')

    return test_loader

