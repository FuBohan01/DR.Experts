import numpy as np
import torch
import torch.distributed as dist
from torchvision import transforms

from .iqa_dataset import *
from .samplers import IQAPatchDistributedSampler, SubsetRandomSampler


def build_transform(is_train, config):
    if config.DATA.DATASET == "koniq":
        if is_train:
            transform = [
                transforms.Compose(
                    [
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomVerticalFlip(),
                        transforms.Resize((512, 384)),
                        transforms.RandomCrop(size=config.DATA.CROP_SIZE),
                    ]
                ),
                transforms.Compose(
                    [
                        transforms.ToTensor(),
                    ]
                ),
                transforms.Compose(
                    [
                        transforms.Normalize(
                            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                        ),
                    ]
                ),
            ]

        else:
            transform = [
                transforms.Compose(
                    [
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomVerticalFlip(),
                        transforms.Resize((512, 384)),
                        transforms.RandomCrop(size=config.DATA.CROP_SIZE),
                    ]
                ),
                transforms.Compose(
                    [
                        transforms.ToTensor(),
                    ]
                ),
                transforms.Compose(
                    [
                        transforms.Normalize(
                            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                        ),
                    ]
                ),
            ]
    elif config.DATA.DATASET == "livec":
        if is_train:
            transform = [
                transforms.Compose(
                    [
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomVerticalFlip(),
                        transforms.RandomCrop(size=config.DATA.CROP_SIZE),
                    ]
                ),
                transforms.Compose(
                    [
                        transforms.ToTensor(),
                    ]
                ),
                transforms.Compose(
                    [
                        transforms.Normalize(
                            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                        ),
                    ]
                ),
            ]
            # Test transforms
        else:
            transform = [
                transforms.Compose(
                    [
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomVerticalFlip(),
                        transforms.RandomCrop(size=config.DATA.CROP_SIZE),
                    ]
                ),
                transforms.Compose(
                    [
                        transforms.ToTensor(),
                    ]
                ),
                transforms.Compose(
                    [
                        transforms.Normalize(
                            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                        ),
                    ]
                ),
            ]
    elif config.DATA.DATASET == "bid":
        if is_train:
            transform = [
                transforms.Compose(
                    [
                        transforms.Resize((512, 512)),
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomVerticalFlip(),
                        transforms.RandomCrop(size=config.DATA.CROP_SIZE),
                    ]
                ),
                transforms.Compose(
                    [
                        transforms.ToTensor(),
                    ]
                ),
                transforms.Compose(
                    [
                        transforms.Normalize(
                            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                        ),
                    ]
                ),
            ]
            # Test transforms
        else:
            transform = [
                transforms.Compose(
                    [
                        transforms.Resize((512, 512)),
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomVerticalFlip(),
                        transforms.RandomCrop(size=config.DATA.CROP_SIZE),
                    ]
                ),
                transforms.Compose(
                    [
                        transforms.ToTensor(),
                    ]
                ),
                transforms.Compose(
                    [
                        transforms.Normalize(
                            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                        ),
                    ]
                ),
            ]
    elif config.DATA.DATASET == "live":
        if is_train:
            transform = [
                transforms.Compose(
                    [
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomVerticalFlip(),
                        transforms.RandomCrop(size=config.DATA.CROP_SIZE),
                    ]
                ),
                transforms.Compose(
                    [
                        transforms.ToTensor(),
                    ]
                ),
                transforms.Compose(
                    [
                        transforms.Normalize(
                            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                        ),
                    ]
                ),
            ]
            # Test transforms
        else:
            transform = [
                transforms.Compose(
                    [
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomVerticalFlip(),
                        transforms.RandomCrop(size=config.DATA.CROP_SIZE),
                    ]
                ),
                transforms.Compose(
                    [
                        transforms.ToTensor(),
                    ]
                ),
                transforms.Compose(
                    [
                        transforms.Normalize(
                            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                        ),
                    ]
                ),
            ]
    elif config.DATA.DATASET == "tid2013":
        if is_train:
            transform = [
                transforms.Compose(
                    [
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomVerticalFlip(),
                        transforms.RandomCrop(size=config.DATA.CROP_SIZE),
                    ]
                ),
                transforms.Compose(
                    [
                        transforms.ToTensor(),
                    ]
                ),
                transforms.Compose(
                    [
                        transforms.Normalize(
                            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                        ),
                    ]
                ),
            ]
            # Test transforms
        else:
            transform = [
                transforms.Compose(
                    [
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomVerticalFlip(),
                        transforms.RandomCrop(size=config.DATA.CROP_SIZE),
                    ]
                ),
                transforms.Compose(
                    [
                        transforms.ToTensor(),
                    ]
                ),
                transforms.Compose(
                    [
                        transforms.Normalize(
                            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                        ),
                    ]
                ),
            ]
    elif config.DATA.DATASET == "csiq":
        if is_train:
            transform = [
                transforms.Compose(
                    [
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomVerticalFlip(),
                        transforms.RandomCrop(size=config.DATA.CROP_SIZE),
                    ]
                ),
                transforms.Compose(
                    [
                        transforms.ToTensor(),
                    ]
                ),
                transforms.Compose(
                    [
                        transforms.Normalize(
                            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                        ),
                    ]
                ),
            ]
            # Test transforms
        else:
            transform = [
                transforms.Compose(
                    [
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomVerticalFlip(),
                        transforms.RandomCrop(size=config.DATA.CROP_SIZE),
                    ]
                ),
                transforms.Compose(
                    [
                        transforms.ToTensor(),
                    ]
                ),
                transforms.Compose(
                    [
                        transforms.Normalize(
                            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                        ),
                    ]
                ),
            ]
    elif config.DATA.DATASET == "kadid":
        if is_train:
            transform = [
                transforms.Compose(
                    [
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomVerticalFlip(),
                        transforms.RandomCrop(size=config.DATA.CROP_SIZE),
                    ]
                ),
                transforms.Compose(
                    [
                        transforms.ToTensor(),
                    ]
                ),
                transforms.Compose(
                    [
                        transforms.Normalize(
                            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                        ),
                    ]
                ),
            ]
            # Test transforms
        else:
            transform = [
                transforms.Compose(
                    [
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomVerticalFlip(),
                        transforms.RandomCrop(size=config.DATA.CROP_SIZE),
                    ]
                ),
                transforms.Compose(
                    [
                        transforms.ToTensor(),
                    ]
                ),
                transforms.Compose(
                    [
                        transforms.Normalize(
                            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                        ),
                    ]
                ),
            ]
    if config.DATA.DATASET == "spaq":
        if is_train:
            transform = [
                transforms.Compose(
                    [
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomVerticalFlip(),
                        transforms.RandomCrop(size=config.DATA.CROP_SIZE),
                    ]
                ),
                transforms.Compose(
                    [
                        transforms.ToTensor(),
                    ]
                ),
                transforms.Compose(
                    [
                        transforms.Normalize(
                            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                        ),
                    ]
                ),
            ]
            # Test transforms
        else:
            transform = [
                transforms.Compose(
                    [
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomVerticalFlip(),
                        transforms.RandomCrop(size=config.DATA.CROP_SIZE),
                    ]
                ),
                transforms.Compose(
                    [
                        transforms.ToTensor(),
                    ]
                ),
                transforms.Compose(
                    [
                        transforms.Normalize(
                            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                        ),
                    ]
                ),
            ]
    if config.DATA.DATASET == "livefb":
        if is_train:
            transform = [
                transforms.Compose(
                    [
                        transforms.Resize((224, 224)),
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomVerticalFlip(),
                        # transforms.RandomCrop(size=config.DATA.CROP_SIZE),
                    ]
                ),
                transforms.Compose(
                    [
                        transforms.ToTensor(),
                    ]
                ),
                transforms.Compose(
                    [
                        transforms.Normalize(
                            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                        ),
                    ]
                ),
            ]
            # Test transforms
        else:
            transform = [
                transforms.Compose(
                    [
                        transforms.Resize((224, 224)),
                        # transforms.RandomHorizontalFlip(),
                        # transforms.RandomVerticalFlip(),
                        # transforms.RandomCrop(size=config.DATA.CROP_SIZE),
                    ]
                ),
                transforms.Compose(
                    [
                        transforms.ToTensor(),
                    ]
                ),
                transforms.Compose(
                    [
                        transforms.Normalize(
                            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                        ),
                    ]
                ),
            ]
    return transform


def build_IQA_dataset(config):
    print(config.DATA.DATASET)
    # if config.DATA.DATASET == "koniq":
    #     train_dataset = KONIQDATASET(
    #         config.DATA.DATA_PATH,
    #         config.SET.TRAIN_INDEX,
    #         config.DATA.PATCH_NUM,
    #         transform=build_transform(is_train=True, config=config),
    #     )
    #     test_dataset = KONIQDATASET(
    #         config.DATA.DATA_PATH,
    #         config.SET.TEST_INDEX,
    #         config.DATA.TEST_PATCH_NUM,
    #         transform=build_transform(is_train=False, config=config),
    #     )
    if config.DATA.DATASET == "koniq":
        train_dataset = KONIQDATASET(
            config.DATA.DATA_PATH,
            config.SET.TRAIN_INDEX,
            config.DATA.PATCH_NUM,
            transform=build_transform(is_train=True, config=config),
        )
        test_dataset = KONIQDATASET(
            config.DATA.DATA_PATH,
            config.SET.TEST_INDEX,
            config.DATA.TEST_PATCH_NUM,
            transform=build_transform(is_train=False, config=config),
        )
    elif config.DATA.DATASET == "bid":
        train_dataset = BIDDATASET(
            config.DATA.DATA_PATH,
            config.SET.TRAIN_INDEX,
            config.DATA.PATCH_NUM,
            transform=build_transform(is_train=True, config=config),
        )
        test_dataset = BIDDATASET(
            config.DATA.DATA_PATH,
            config.SET.TEST_INDEX,
            config.DATA.TEST_PATCH_NUM,
            transform=build_transform(is_train=False, config=config),
        )
    elif config.DATA.DATASET == "uw":
        train_dataset = UWIQADATASET(
            config.DATA.DATA_PATH,
            config.SET.TRAIN_INDEX,
            config.DATA.PATCH_NUM,
            transform=build_transform(is_train=True, config=config),
        )
        test_dataset = UWIQADATASET(
            config.DATA.DATA_PATH,
            config.SET.TEST_INDEX,
            config.DATA.PATCH_NUM,
            transform=build_transform(is_train=False, config=config),
        )
    elif config.DATA.DATASET == "livec":
        train_dataset = LIVECDATASET(
            config.DATA.DATA_PATH,
            config.SET.TRAIN_INDEX,
            config.DATA.PATCH_NUM,
            transform=build_transform(is_train=True, config=config),
        )
        test_dataset = LIVECDATASET(
            config.DATA.DATA_PATH,
            config.SET.TEST_INDEX,
            config.DATA.TEST_PATCH_NUM,
            transform=build_transform(is_train=False, config=config),
        )
    elif config.DATA.DATASET == "live":
        train_dataset = LIVEDataset(
            config.DATA.DATA_PATH,
            config.SET.TRAIN_INDEX,
            config.DATA.PATCH_NUM,
            transform=build_transform(is_train=True, config=config),
        )
        test_dataset = LIVEDataset(
            config.DATA.DATA_PATH,
            config.SET.TEST_INDEX,
            config.DATA.TEST_PATCH_NUM,
            transform=build_transform(is_train=False, config=config),
        )
    elif config.DATA.DATASET == "tid2013":
        train_dataset = TID2013Dataset(
            config.DATA.DATA_PATH,
            config.SET.TRAIN_INDEX,
            config.DATA.PATCH_NUM,
            transform=build_transform(is_train=True, config=config),
        )
        test_dataset = TID2013Dataset(
            config.DATA.DATA_PATH,
            config.SET.TEST_INDEX,
            config.DATA.TEST_PATCH_NUM,
            transform=build_transform(is_train=False, config=config),
        )
    elif config.DATA.DATASET == "csiq":
        train_dataset = CSIQDataset(
            config.DATA.DATA_PATH,
            config.SET.TRAIN_INDEX,
            config.DATA.PATCH_NUM,
            transform=build_transform(is_train=True, config=config),
        )
        test_dataset = CSIQDataset(
            config.DATA.DATA_PATH,
            config.SET.TEST_INDEX,
            config.DATA.TEST_PATCH_NUM,
            transform=build_transform(is_train=False, config=config),
        )
    elif config.DATA.DATASET == "kadid":
        train_dataset = KADIDDataset(
            config.DATA.DATA_PATH,
            config.SET.TRAIN_INDEX,
            config.DATA.PATCH_NUM,
            transform=build_transform(is_train=True, config=config),
        )
        test_dataset = KADIDDataset(
            config.DATA.DATA_PATH,
            config.SET.TEST_INDEX,
            config.DATA.TEST_PATCH_NUM,
            transform=build_transform(is_train=False, config=config),
        )
    elif config.DATA.DATASET == "spaq":
        train_dataset = SPAQDATASET(
            config.DATA.DATA_PATH,
            config.SET.TRAIN_INDEX,
            config.DATA.PATCH_NUM,
            transform=build_transform(is_train=True, config=config),
        )
        test_dataset = SPAQDATASET(
            config.DATA.DATA_PATH,
            config.SET.TEST_INDEX,
            config.DATA.TEST_PATCH_NUM,
            transform=build_transform(is_train=False, config=config),
        )
    elif config.DATA.DATASET == "livefb":
        train_dataset = FBLIVEFolder(
            config.DATA.DATA_PATH,
            config.SET.TRAIN_INDEX,
            config.DATA.PATCH_NUM,
            transform=build_transform(is_train=True, config=config),
        )
        test_dataset = FBLIVEFolder(
            config.DATA.DATA_PATH,
            config.SET.TEST_INDEX,
            config.DATA.TEST_PATCH_NUM,
            transform=build_transform(is_train=False, config=config),
        )
    else:
        raise NotImplementedError("We only support common IQA dataset Now.")

    return train_dataset, test_dataset


def IQA_build_loader(config):
    config.defrost()
    dataset_train, dataset_val = build_IQA_dataset(config=config)
    config.freeze()
    print(
        f"local rank {config.LOCAL_RANK} / global rank {dist.get_rank()} successfully build train dataset"
    )
    print(
        f"local rank {config.LOCAL_RANK} / global rank {dist.get_rank()} successfully build val dataset"
    )

    num_tasks = dist.get_world_size()
    global_rank = dist.get_rank()
    if config.DATA.ZIP_MODE and config.DATA.CACHE_MODE == "part":
        indices = np.arange(dist.get_rank(), len(dataset_train), dist.get_world_size())
        sampler_train = SubsetRandomSampler(indices)
    else:
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )

    if config.TEST.SEQUENTIAL:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_val = IQAPatchDistributedSampler(dataset_val)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=config.DATA.BATCH_SIZE,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        # collate_fn=orders_collate_fn,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        sampler=sampler_val,
        batch_size=config.DATA.BATCH_SIZE,
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        # collate_fn=orders_collate_fn,
        drop_last=False,
    )

    return dataset_train, dataset_val, data_loader_train, data_loader_val


def orders_collate_fn(batch):
    images = []
    depths = []
    orders = []
    targets = []
    score_ts = []

    for sample in batch:
        image, depth, order, target, score_t = sample
        images.append(image)
        depths.append(depth)
        orders.append(order)
        targets.append(target)
        score_ts.append(score_t)

    images = torch.stack(images, dim=0)
    depths = torch.stack(depths, dim=0)
    score_ts = torch.stack(score_ts, dim=0)

    targets = torch.tensor(targets)

    return images, depths, orders, targets, score_ts
