import numpy as np
import torch
from skimage.transform import resize
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

import cfg
import utils


class ImageDataset(Dataset):
    def __init__(self, image_paths: np.ndarray, mask_paths: np.ndarray, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return self.image_paths.shape[0]

    def __getitem__(self, index):
        p_name = self.image_paths[index].stem

        image = utils.read_mha(self.image_paths[index], norm=(0., 1.), dtype=np.float32)

        mask = utils.read_mha(self.mask_paths[index])
        # convert to PyTorch order: (C x H x W)
        # mask = np.transpose(mask, (2, 0, 1))  # transforms.ToTensor() do this
        # 1only batches of spatial targets supported
        mask = np.squeeze(mask, axis=-1)

        # resize
        # image = resize(image, (128, 128), mode='reflect').astype(np.float32)
        # mask = resize(mask, (128, 128), mode='reflect').astype(np.long)

        if self.transform:
            image = self.transform(image)

        return image, mask, p_name


def image_dataloader(splited):
    X_train, X_test, y_train, y_test = splited

    train_dataloader = DataLoader(ImageDataset(X_train, y_train, transform=transforms.Compose([
        transforms.ToTensor()
    ])), batch_size=cfg.batch_size, shuffle=True)

    val_dataloader = DataLoader(ImageDataset(X_test, y_test, transform=transforms.Compose([
        transforms.ToTensor()
    ])), batch_size=cfg.batch_size, shuffle=False)

    return train_dataloader, val_dataloader


def get_all_data_path(selected_list=None):
    path_list = list(cfg.ori_dir.iterdir())
    path_list = np.array(sorted(path_list))

    if not selected_list:
        if not cfg.n_samples:
            n_samples = len(path_list)
        else:
            n_samples = cfg.n_samples
        selected_list = range(n_samples)
    path_list = path_list[selected_list]

    ori_paths = []
    seg_paths = []
    n_ori = len(path_list)
    for i, ori_path in enumerate(path_list, start=1):
        print(f'[{i}/{n_ori}] {ori_path.stem}')

        seg_path = cfg.seg_dir.joinpath(ori_path.stem + '_seg' + ori_path.suffix)

        ori_paths.append(ori_path)
        seg_paths.append(seg_path)

    return np.asarray(ori_paths), np.asarray(seg_paths)
