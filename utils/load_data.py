import sys
sys.path.append("..")
import os
import warnings
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch import nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import random
import numpy as np
import deeplake
import matplotlib
matplotlib.use('Agg')

SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

def load_PACS(batch_size):
    ds_train = deeplake.load("./dataset/PACS/pacs-train-local", read_only=True, verbose=False)
    ds_test = deeplake.load("./dataset/PACS/pacs-test-local", read_only=True, verbose=False)
    print("Dataset loaded successfully.")

    transform_train = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    transform_test = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    def get_domain_indices(dataset):
        domain_indices = {0: [], 1: [], 2: [], 3: []}

        for idx, (img, label, domain) in enumerate(zip(dataset.images, dataset.labels, dataset.domains)):
            domain = domain.data()['value'][0]
            domain_indices[domain].append(idx)

        return domain_indices

    def align_number_of_datasets(dict_merged):
        dict_merged_list_lengths = {key: len(value) for key, value in dict_merged.items()}
        min_len_client_dataset = min(dict_merged_list_lengths.values())
        for key, value in dict_merged.items():
            if len(value) > min_len_client_dataset:
                dict_merged[key] = list(sorted(random.sample(value, min_len_client_dataset)))
        return dict_merged

    def collate_fn(batch, batch_size):
        current_batch_size = len(batch)

        if current_batch_size < batch_size:
            padding_size = batch_size - current_batch_size
            padded_data = batch + [(torch.zeros_like(batch[0][0]), torch.zeros_like(batch[0][1]))] * padding_size
            batch = padded_data

        return torch.utils.data.dataloader.default_collate(batch)

    class CustomSubset(Dataset):
        def __init__(self, dataset, indices, transform=None):
            self.dataset = dataset
            self.indices = indices
            self.transform = transform
            self.images = []
            self.labels = []

            for idx in self.indices:
                data = self.dataset[idx]
                img = data.images.data()['value']
                label = data.labels.data()['value'][0]
                if isinstance(img, np.ndarray):
                    img = Image.fromarray(img)
                self.images.append(img)
                self.labels.append(label)

            self.labels = [torch.tensor(l, dtype=torch.long) for l in self.labels]

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, idx):
            img = self.images[idx]
            label = self.labels[idx]
            if self.transform:
                img = self.transform(img)
            return img, label

    domain_indices_train = get_domain_indices(ds_train)
    domain_indices_test = get_domain_indices(ds_test)
    print("Got domain indices.")

    domain_indices_train = align_number_of_datasets(domain_indices_train)

    large_domains = [0, 1]
    small_domains = [2, 3]

    client_indices = {}
    client_id = 0

    for domain, indices in domain_indices_train.items():
        labels = [ds_train[i].labels.data()['value'][0] for i in indices]
        label_to_indices = {}
        for idx, label in zip(indices, labels):
            label_to_indices.setdefault(label, []).append(idx)

        split_num = 3 if domain in large_domains else 2
        domain_split = {i: [] for i in range(split_num)}

        for label, idxs in label_to_indices.items():
            random.shuffle(idxs)
            split_sizes = [len(idxs) // split_num] * split_num
            for i in range(len(idxs) % split_num):
                split_sizes[i] += 1
            start = 0
            for i in range(split_num):
                domain_split[i].extend(idxs[start:start + split_sizes[i]])
                start += split_sizes[i]

        for i in range(split_num):
            client_indices[client_id] = sorted(domain_split[i])
            client_id += 1

    merged_train_loaders = {
        cid: DataLoader(
            CustomSubset(ds_train, indices=idxs, transform=transform_train),
            batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True,
            collate_fn=lambda batch: collate_fn(batch, batch_size=batch_size)
        )
        for cid, idxs in client_indices.items()
    }
    print("Created training dataloaders (10 clients).")

    test_indices = []
    for indices in domain_indices_test.values():
        test_indices.extend(indices)

    test_loader = DataLoader(
        CustomSubset(ds_test, indices=test_indices, transform=transform_test),
        batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )

    test_loaders_for_domains = {
        domain: DataLoader(
            CustomSubset(ds_test, indices=indices, transform=transform_test),
            batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
        )
        for domain, indices in domain_indices_test.items()
    }

    print("Test set size:", len(test_indices))
    for cid in merged_train_loaders:
        print(f"Client {cid}: {len(client_indices[cid])} samples")

    return merged_train_loaders, test_loader, client_indices, test_indices, test_loaders_for_domains, domain_indices_test


from torch.utils.data import DataLoader, Subset, ConcatDataset, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder

def load_officehome(data_root='./dataset/OfficeHome/dataset', batch_size=64, train_split=0.8):
    print("Loading OfficeHome datasets...")

    transform_train = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))
    ])

    transform_test = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))
    ])

    def collate_fn(batch, batch_size):
        current_batch_size = len(batch)
        if current_batch_size < batch_size:
            padding_size = batch_size - current_batch_size
            image_pad = torch.zeros_like(batch[0][0])
            label_pad = torch.tensor(0, dtype=torch.long)
            padded_data = batch + [(image_pad, label_pad)] * padding_size
            batch = padded_data
        return torch.utils.data.dataloader.default_collate(batch)

    domains = ['Art', 'Clipart', 'Product', 'Real_World']
    large_domains = ['Product', 'Real_World']
    small_domains = ['Art', 'Clipart']

    full_train_loaders = {}
    dict_users_train = {}
    client_id = 0

    train_sets = {}
    test_sets = {}

    for domain in domains:
        domain_path = os.path.join(data_root, domain)
        if not os.path.exists(domain_path):
            print(f"[WARN] Directory {domain_path} does not exist.")
            continue

        full_dataset = ImageFolder(domain_path)

        indices = list(range(len(full_dataset)))
        random.shuffle(indices)
        train_size = int(train_split * len(indices))
        train_indices = indices[:train_size]
        test_indices = indices[train_size:]

        all_labels = full_dataset.targets
        train_labels = [all_labels[i] for i in train_indices]

        label_to_indices = {}
        for idx, label in zip(train_indices, train_labels):
            label_to_indices.setdefault(label, []).append(idx)

        split_count = 3 if domain in large_domains else 2
        sub_domain_indices = {i: [] for i in range(split_count)}

        for label, label_idxs in label_to_indices.items():
            random.shuffle(label_idxs)
            split_sizes = [len(label_idxs) // split_count] * split_count
            for i in range(len(label_idxs) % split_count):
                split_sizes[i] += 1
            start = 0
            for i in range(split_count):
                sub_domain_indices[i].extend(label_idxs[start:start + split_sizes[i]])
                start += split_sizes[i]

        for i in range(split_count):
            subset = Subset(ImageFolder(domain_path, transform=transform_train), sub_domain_indices[i])
            full_train_loaders[client_id] = DataLoader(
                subset, batch_size=batch_size, shuffle=True, num_workers=4,
                collate_fn=lambda batch: collate_fn(batch, batch_size=batch_size)
            )
            dict_users_train[client_id] = sub_domain_indices[i]
            print(f"[Train] Domain {domain}, Split {i}, Client {client_id}: {len(sub_domain_indices[i])} samples")
            client_id += 1

        test_set = Subset(ImageFolder(domain_path, transform=transform_test), test_indices)
        test_sets[domain] = test_set

    test_loaders = {
        domain: DataLoader(
            ds, batch_size=batch_size, shuffle=False, num_workers=4
        ) for domain, ds in test_sets.items()
    }

    test_loader = DataLoader(
        ConcatDataset(list(test_sets.values())),
        batch_size=batch_size, shuffle=False, num_workers=4
    )

    return full_train_loaders, test_loader, test_loaders