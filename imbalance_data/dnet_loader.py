# original code: https://github.com/frank-xwang/RIDE-LongTailRecognition/blob/main/data_loader/imagenet_lt_data_loaders.py


import os
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
from collections import Counter
import random

class BaseLoader(Dataset):

    def __init__(self, root, txt, transform=None, use_randaug=False, weighted_alpha=1, 
                        imb_type='exp', imb_factor=1., cls_num=100):
        self.img_path = []
        self.labels = []
        self.transform = transform
        self.use_randaug = use_randaug
        self.cls_num = cls_num
        with open(txt) as f:
            for line in f:
                self.img_path.append(os.path.join(root, line.split()[0]))
                self.labels.append(int(line.split()[1]))
        self.targets = self.labels  # Sampler needs to use targets

        n_per_cls = self.get_cls_num_list()
        self.weighted_alpha = weighted_alpha
        if imb_factor != 1:
            img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type, imb_factor, n_per_cls)
            self.gen_imbalanced_data(img_num_list)
        self.use_randaug = use_randaug

    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor, n_per_cls=[]):
        img_max = max(n_per_cls)# len(self.data) / cls_num
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor ** (cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        return img_num_per_cls

    def gen_imbalanced_data(self, img_num_per_cls):
        new_data = []
        new_targets = []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)
        # np.random.shuffle(classes)

        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            # print(selec_idx)
            new_data.append(self.data[selec_idx, ...])
            new_targets.extend([the_class, ] * the_img_num)
        new_data = np.vstack(new_data)
        self.data = new_data
        self.targets = new_targets

    def get_cls_num_list(self):
        # cls_num_list = []
        # for i in range(self.cls_num):
        #     cls_num_list.append(self.num_per_cls_dict[i])
        # return cls_num_list

        counter = Counter(self.targets)
        n_per_cls = []
        for i in range(self.cls_num):
            n_per_cls.append(counter.get(i, 1e-7))

        return n_per_cls

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):

        path = self.img_path[index]
        label = self.labels[index]

        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')

        if self.use_randaug:
            r = random.random()
            if r < 0.5:
                sample = self.transform[0](sample)
            else:
                sample = self.transform[1](sample)
        else:
            if self.transform is not None:
                sample = self.transform(sample)

        # return sample, label, path
        return sample, label

    def get_weighted_sampler(self):
        cls_num_list = self.get_cls_num_list()
        cls_weight = 1.0 / (np.array(cls_num_list) ** self.weighted_alpha)
        cls_weight = cls_weight / np.sum(cls_weight) * len(cls_num_list)
        samples_weight = np.array([cls_weight[t] for t in self.targets])
        samples_weight = torch.from_numpy(samples_weight)
        samples_weight = samples_weight.double()
        print("samples_weight", samples_weight)
        sampler = torch.utils.data.WeightedRandomSampler(samples_weight, len(self.targets), replacement=True)
        return sampler

