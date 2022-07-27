# -*- coding: utf-8 -*-

from logging import root
import os
from re import S 
import torch
import random
import h5py 
import pandas as pd
from scipy import ndimage
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler

class H5DataSets(Dataset):
    """
    Dataset for loading images stored in h5 format. It generates 
    4D tensors with dimention order [C, D, H, W] for 3D images, and 
    3D tensors with dimention order [C, H, W] for 2D images
    """
    def __init__(self, root_dir, sample_list_name, transform = None):
        self.root_dir = root_dir 
        self.transform  = transform
        with open(sample_list_name, 'r') as f:
            lines = f.readlines()
            self.sample_list = [item.replace('\n', '') for item in lines]

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        sample_name = self.sample_list[idx]
        h5f = h5py.File(self.root_dir + '/' +  sample_name, 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        # sample["idx"] = idx
        return sample
    
class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """

    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in zip(grouper(primary_iter, self.primary_batch_size),
                   grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)


if __name__ == "__main__":
    root_dir = "/home/guotai/disk2t/projects/semi_supervise/SSL4MIS/data/ACDC/data/slices"
    file_name = "/home/guotai/disk2t/projects/semi_supervise/slices.txt"
    dataset = H5DataSets(root_dir, file_name)
    train_loader = torch.utils.data.DataLoader(dataset, 
                batch_size = 4, shuffle=True, num_workers= 1)
    for sample in train_loader:
        image = sample['image']
        label = sample['label']
        print(image.shape, label.shape)
        print(image.min(), image.max(), label.max())