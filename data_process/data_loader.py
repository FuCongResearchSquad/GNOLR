import json
import torch
import numpy as np
from os import path
from functools import partial
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, concatenate_datasets


def custom_collate_fn_list(batch):
    """
    return:
    - result_tensor: [items, features] All data (including labels).
    - labels_tensor: [items, task]
    - list(sizes): The number of items each user has interacted with
    """
    result_tensors, labels, sizes = zip(*batch)
    result_tensor = torch.cat(result_tensors, dim=0)
    labels_tensor = torch.cat(labels, dim=0)

    return result_tensor, labels_tensor, list(sizes)


def custom_collate_fn(batch):
    """
    return:
    - result_tensor: [items, features] All data (including labels).
    - labels_tensor: [items, task]
    - list(sizes): 1
    """
    result_tensors, labels, sizes = zip(*batch)
    result_tensor = torch.cat(result_tensors, dim=0)
    labels_tensor = torch.cat(labels, dim=0)

    return result_tensor, labels_tensor, list(sizes)


def get_data_feature(dir):
    with open(path.join(dir, "feature.json"), "r") as file:
        feature_data = json.load(file)
    item_feature = [entry for entry in feature_data if entry["belongs"] == "item"]
    user_feature = [entry for entry in feature_data if entry["belongs"] == "user"]
    label_feature = [entry for entry in feature_data if entry["belongs"] == "label"]
    return user_feature, item_feature, label_feature, feature_data


class CustomDatasetList(Dataset):
    def __init__(self, dataset, feature_data):
        self.dataset = dataset
        self.feature_data = feature_data

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        The data is read as a single line containing a user and the items they have interacted with, with each item separated by '\x01'.
        This function converts the single line into multiple lines, where each line represents a user-item interaction record.
        """
        example = self.dataset[idx]
        idx = []
        rows = [value.split("\x01") for value in example.values()]
        # The length is the number of items the current user has interacted with
        length = len(rows[0])
        for item in self.feature_data:
            if item["belongs"] == "label":
                idx.append(item["index"])

        result_tensor = torch.from_numpy(np.array(rows).astype(np.float32)).T
        return result_tensor, result_tensor[:, idx], length


class CustomDataset(Dataset):
    def __init__(self, dataset, feature_data):
        self.dataset = dataset
        self.feature_data = feature_data

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]
        idx = []
        rows = [value for value in example.values()]
        length = 1
        for item in self.feature_data:
            if item["belongs"] == "label":
                idx.append(item["index"])

        result_tensor = torch.from_numpy(np.array(rows).astype(np.float32)).T
        result_tensor = result_tensor.unsqueeze(0)
        return result_tensor, result_tensor[:, idx], length


def data_loader(
    dir: str,
    batch_size: int,
    data_loader_worker: int,
    is_list: str = "true",
    CACHE_DIR: str = "./cache",
):
    """
    Sample:
        batch=2, is_list=false: [ [user_1, item_1],
                                  [user_1, item_2] ]
        batch=2, is_list=true:  [ [user_1, [item_1, item_2, item_3]],
                                  [user_2, [item_1, item_2]] ]
    """

    if is_list.lower() == "false":
        train_path = path.join(dir, "train", "point", "*.parquet")
        test_path = path.join(dir, "test", "point", "*.parquet")
    elif is_list.lower() == "true":
        train_path = path.join(dir, "train", "list", "*.parquet")
        test_path = path.join(dir, "test", "list", "*.parquet")
    else:
        raise ValueError(f"Unknown is_list {is_list}")

    dataset = load_dataset(
        "parquet",
        data_files={"train": train_path, "test": test_path},
        cache_dir=CACHE_DIR,
    )

    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    user_feature, item_feature, label_feature, feature_data = get_data_feature(dir)

    if is_list.lower() == "false":
        train_dataset = CustomDataset(train_dataset, feature_data)
        test_dataset = CustomDataset(test_dataset, feature_data)
    elif is_list.lower() == "true":
        train_dataset = CustomDatasetList(train_dataset, feature_data)
        test_dataset = CustomDatasetList(test_dataset, feature_data)

    valid_size = int(0.1 * len(train_dataset))
    train_size = len(train_dataset) - valid_size
    train_dataset, valid_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, valid_size]
    )

    if is_list.lower() == "false":
        collate_fn_with_args = partial(custom_collate_fn)
    elif is_list.lower() == "true":
        collate_fn_with_args = partial(custom_collate_fn_list)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=data_loader_worker,
        collate_fn=collate_fn_with_args,
    )

    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=data_loader_worker,
        collate_fn=collate_fn_with_args,
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=data_loader_worker,
        collate_fn=collate_fn_with_args,
    )

    return (
        train_dataloader,
        valid_dataloader,
        test_dataloader,
        user_feature,
        item_feature,
        label_feature,
    )


def custom_recall_collate_fn(batch):
    result_tensors, labels, sample_ids, sizes = zip(*batch)

    result_tensor = torch.cat(result_tensors, dim=0)
    labels_tensor = torch.cat(labels, dim=0)
    sample_ids_tensor = torch.cat(sample_ids, dim=0)

    return result_tensor, labels_tensor, sample_ids_tensor, list(sizes)


class CustomRecallDatasetKR(Dataset):
    def __init__(self, dataset, feature_data):
        self.dataset = dataset
        self.feature_data = feature_data

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        The data is read as a single line containing a user and the items they have interacted with, with each item separated by '\x01'.
        This function converts the single line into multiple lines, where each line represents a user-item interaction record.
        """
        example = self.dataset[idx]

        label_idx = []
        sampleid_idx = []

        rows = [value.split("\x01") for value in example.values()]
        length = len(rows[0])
        for item in self.feature_data:
            if item["belongs"] == "label":
                label_idx.append(item["index"])
            if item["name"] == "user_id":
                sampleid_idx.append(item["index"])
            if item["name"] == "video_id":
                sampleid_idx.append(item["index"])

        result_tensor = torch.from_numpy(np.array(rows).astype(np.float32)).T
        return (
            result_tensor,
            result_tensor[:, label_idx],
            result_tensor[:, sampleid_idx],
            length,
        )


class CustomRecallDatasetML(Dataset):
    def __init__(self, dataset, feature_data):
        self.dataset = dataset
        self.feature_data = feature_data

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        The data is read as a single line containing a user and the items they have interacted with, with each item separated by '\x01'.
        This function converts the single line into multiple lines, where each line represents a user-item interaction record.
        """
        example = self.dataset[idx]

        label_idx = []
        sampleid_idx = []
        rows = [value.split("\x01") for value in example.values()]
        length = len(rows[0])
        for item in self.feature_data:
            if item["belongs"] == "label":
                label_idx.append(item["index"])
            if item["name"] == "user_id":
                sampleid_idx.append(item["index"])
            if item["name"] == "movie_id":
                sampleid_idx.append(item["index"])

        result_tensor = torch.from_numpy(np.array(rows).astype(np.float32)).T
        return (
            result_tensor,
            result_tensor[:, label_idx],
            result_tensor[:, sampleid_idx],
            length,
        )


def recall_data_loader(
    dir: str,
    batch_size: int,
    data_loader_worker: int,
    embedding_type: str,
    is_list: str = "true",
    DATASET_TYPE: str = "KR",
    CACHE_DIR: str = "./cache",
):

    if is_list.lower() == "false":
        train_path = path.join(dir, "train", "point", "*.parquet")
        test_path = path.join(dir, "test", "point", "*.parquet")
    elif is_list.lower() == "true":
        train_path = path.join(dir, "train", "list", "*.parquet")
        test_path = path.join(dir, "test", "list", "*.parquet")
    else:
        raise ValueError(f"Unknown is_list {is_list}")

    dataset = load_dataset(
        "parquet",
        data_files={"train": train_path, "test": test_path},
        cache_dir=CACHE_DIR,
    )

    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    # user only for test dateset && item for all train/test dataset
    if embedding_type == "user":
        print(
            f"[RecallExp]RecallDataLoader [EmbeddingType] {embedding_type}, ONLY TEST DATASET"
        )
        combined_dataset = test_dataset
    else:
        assert embedding_type == "item"
        if DATASET_TYPE.lower() == "kr":
            print(
                f"[RecallExp]RecallDataLoader [EmbeddingType] {embedding_type}, ONLY TEST DATASET"
            )
            combined_dataset = test_dataset
        else:
            assert DATASET_TYPE.lower() == "ml"
            print(
                f"[RecallExp]RecallDataLoader [EmbeddingType] {embedding_type}, ALL TRAIN & TEST DATASET"
            )
            combined_dataset = concatenate_datasets([train_dataset, test_dataset])

    user_feature, item_feature, label_feature, feature_data = get_data_feature(dir)

    if DATASET_TYPE.lower() == "kr":
        test_dataset = CustomRecallDatasetKR(combined_dataset, feature_data)
    else:
        assert DATASET_TYPE.lower() == "ml"
        test_dataset = CustomRecallDatasetML(combined_dataset, feature_data)

    collate_fn_with_args = partial(custom_recall_collate_fn)

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=data_loader_worker,
        collate_fn=collate_fn_with_args,
    )

    return None, test_dataloader, user_feature, item_feature, label_feature
