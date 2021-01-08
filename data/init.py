import torch.utils.data as Data
from PIL import Image
import torch
from transfrom import get_transform

import copy
import random
from collections import defaultdict

import numpy as np
from torch.utils.data.sampler import Sampler


class ImageDataset(Data.Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid = self.dataset[index]
        img = Image.open(img_path)
        if self.transform is not None:
            img = self.transform(img)
        return img, pid, camid, img_path


def train_collate_fn(batch):
    imgs, pids, _, _, = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids


def val_collate_fn(batch):
    imgs, pids, camids, _ = zip(*batch)
    return torch.stack(imgs, dim=0), pids, camids


'''
Reid usually uses metric learning, and metric learning usually requires positive and negative samples.
To ensure that there are positive and negative samples in a batch,
we take N different pids.  Between pid and pid are negative samples, 
and each pid takes K different samples Photos, between these photos is a positive sample.
'''
class RandomIdentitySampler(Sampler):
    def __init__(self, data_source, batch_size, num_instances):
        self.data_source = data_source
        self.batch_size = batch_size

        '''K instances'''
        self.num_instances = num_instances

        '''N pids'''
        self.num_pids_per_batch = self.batch_size // self.num_instances

        self.index_dic = defaultdict(list)
        for index, (_, pid, _) in enumerate(self.data_source):
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())

        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)
        for pid in self.pids:

            '''choose k instances'''
            idxs = copy.deepcopy(self.index_dic[pid])
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
            random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []
        '''choose N pid and pid's K instances'''
        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)

        self.length = len(final_idxs)
        return iter(final_idxs)

    def __len__(self):
        return self.length






def make_data_loader(datasets,cfg):
    train_transform=get_transform(cfg,training=True)
    test_transform=get_transform(cfg,training=False)
    if cfg['Multi_Data']:
        data=datasets[0]
        for dataset in datasets[1:]:
            data.train.extend(dataset.train)
            data.query.extend(dataset.query)
            data.gallery.extend(dataset.gallery)

            data.num_train_pids+=dataset.num_train_pids
    else:
        data=datasets

    num_classes=data.num_train_pids
    train_set=ImageDataset(data.train,train_transform)

    if cfg['SAMPLER']=='softmax':
        train_loader=Data.DataLoader(train_set,batch_size=cfg['train_bs'],
                                     shuffle=True,collate_fn=train_collate_fn)
    else:
        train_loader=Data.DataLoader(train_set,batch_size=cfg['train_bs'],
                                     sampler=RandomIdentitySampler(data.train,
                                    cfg['train_bs'],cfg['train_K_instances']),
                                     collate_fn=train_collate_fn)


    '''[:num_query] are query data, [num_quer:] are gallery data'''
    val_set=ImageDataset(dataset.query+data.gallery,test_transform)
    val_loader=Data.DataLoader(val_set,batch_size=cfg['test_bs'],shuffle=False,
                               collate_fn=val_collate_fn)
    return train_loader,val_loader,len(data.query),num_classes


