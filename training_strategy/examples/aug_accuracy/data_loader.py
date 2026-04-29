# Code is copied and adapted from here
# https://github.com/ahendriksen/msd_pytorch/blob/master/msd_pytorch/image_dataset.py
#from turtle import pd
import numpy as np
from pathlib import Path
import imageio
import random
import logging
import re
import torch
from torch.utils.data import Dataset
import pandas as pd


def _natural_sort(l):
    def key(x):
        return [int(c) if c.isdigit() else c for c in re.split("([0-9]+)", x)]

    return sorted(l, key=key)


class InputError(Exception):
    def __init__(self, message):
        self.message = message


def avocado_binary_classification(gt_fname):
    thr = 10 ** -2
    gt = np.loadtxt(gt_fname, skiprows=1, delimiter=",")
    air_ratio = gt[:, 4] / gt[:, 1:].sum(axis=1)
    y = np.where(air_ratio > thr, 1, 0)
    return y


def playdoh_triple_classification(gt_fname):
    gt = np.loadtxt(gt_fname, skiprows=1, delimiter=",")
    print("gt shape:", gt.shape)
    print("gt contents:", gt)
    if len(gt.shape) == 1:
        gt = gt.reshape(1, -1)  # 确保 gt 是二维数组
    stone_count = gt[:, 3]
    y = stone_count.astype(int)
    print("stone_count:", stone_count)
    return y


class_func_dict = {
    'avocado_binary_classification': avocado_binary_classification,
    'playdoh_triple_classification': playdoh_triple_classification
}


class ImageStack(object):
    def __init__(self, path_specifier, *, collapse_channels=False, labels=None):#
        self.path_specifier = path_specifier#
        self.collapse_channels = collapse_channels#
        self.labels = labels#
        self.paths = self._read_paths_from_file(path_specifier)  # 读取图像路径

    def _read_paths_from_file(self, file_name):
        with open(file_name, 'r') as f:
            return [line.strip() for line in f]  # 读取每行作为路径

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        path = self.paths[i]
        try:
            img = imageio.v2.imread(path)
        except Exception as e:
            raise logging.exception("Could not read image from {}".format(path))

        if img.ndim != 2:
            raise InputError("Tif image is supposed to have only one channel: {}".format(path))

        img = torch.from_numpy(img)

        return img

    def find_images(self, specifier):
        if isinstance(specifier, str):
            specifier = Path(specifier)#如果输入 specifier 是一个字符串，将其转换为 Path 对象。
        if not isinstance(specifier, (str, Path)):
            raise TypeError(f"Expected str or Path, got {type(specifier)}")
        paths = []
        if specifier.is_dir():
            subfolders = [f for f in specifier.iterdir() if f.is_dir()]  # 遍历子文件夹
            for subfolder in subfolders:
                log_folder = subfolder / 'log'  # 查找 'log' 文件夹
                if log_folder.is_dir():
                    paths.extend(list(log_folder.glob("*.tiff")))  # 添加所有 .tiff 文件路径
        else:
            raise InputError("Expected a directory path, got {}".format(specifier))

        paths = [str(p) for p in paths]
        paths = _natural_sort(paths)  # 自然排序路径
        if len(paths) == 0:
            logging.warning("Image stack is empty for path specification {}".format(specifier))
        return paths



class MultiImageStack(object):
    def __init__(self, path_specifier, *, collapse_channels=False, labels=None):
        self.path_specifier = Path(path_specifier).expanduser().resolve()
        self.collapse_channels = collapse_channels
        self.labels = labels
        self.paths = self._read_paths_from_file(path_specifier)  # 读取图像路径

    def _read_paths_from_file(self, file_name):
        with open(file_name, 'r') as f:
            return [line.strip() for line in f]  # 读取每行作为路径

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        path = self.paths[i]
        try:
            img = imageio.mimread(path)
        except Exception as e:
            raise logging.exception("Could not read image from {}".format(path))

        img = np.array(img, dtype=np.float32)
        img = torch.from_numpy(img)

        return img
    def is_file(self):
        pass

    def is_dir(self):
        pass


class ImageDatasetTransformable(Dataset):
    """Adaptation of Image Dataset from msd_pytorch with standard augmentation (random crop, flips, rotation).
    All augmentation are implemented using standard tensor operations, so it should work with torchvision 0.2.2
    """

    def __init__(self, input_paths_file, target_labels_file, data_conf, *,
                 random_crop=False, padding=20, crop_shape=(380, 478), vertical_flip=False, horizontal_flip=False,
                 rotate=False, filter_interval=None):
        super(ImageDatasetTransformable, self).__init__()
        self.filtered_indices = None
        self.random_crop = random_crop
        self.padding = padding
        self.crop_shape = crop_shape
        self.vertical_flip = vertical_flip
        self.horizontal_flip = horizontal_flip
        self.rotate = rotate
        self.input_paths = self._read_paths_from_file(input_paths_file)  # 读取图像路径
        self.target_labels = self._load_target_labels(target_labels_file)  # 读取标签
        self.filter_interval = filter_interval
        print(f"处理间隔: {filter_interval}.")

        if len(self.input_paths) != len(self.target_labels):
            raise InputError(
                "Number of inputs and target labels does not match. Got {} inputs and {} target labels.".format(len(self.input_paths), len(self.target_labels)))

    # 初始化输入数据栈
        self.input_stack = self._create_input_stack(input_paths_file, data_conf)
        if self.filter_interval is not None:
            self.filtered_indices = list(range(0, len(self.input_stack), self.filter_interval))
            #filter_interval 设置为一个整数n，那么代码会在 input_stack 中每隔 n 个元素选择一个数据点。
        else:
            self.filtered_indices = list(range(len(self.input_stack)))

    def _create_input_stack(self, path_file, data_conf):
        if data_conf['c_in'] > 1:
            return MultiImageStack(path_file)
        elif data_conf['c_in'] == 1:
            return ImageStack(path_file)
        else:
            raise InputError(
                "Number of input channels should be a positive integer. Got {}".format(data_conf['c_in']))


    def _load_target_labels(self, target_labels):
        df = pd.read_csv(target_labels, delimiter=',', header=None, names=['path', 'label'])
        # 提取图片路径和标签
        paths = df['path'].tolist()
        labels = df['label'].astype(int).values  # 确保标签为整数

        # 确保路径和标签数量匹配
        if len(paths) != len(labels):
            raise InputError(
                "Number of paths and labels does not match. Got {} paths and {} labels.".format(len(paths), len(labels)))
        #print(f"Loaded labels: {labels[:5]}对应{labels[:5]}")  # 打印前10个标签进行检查
        return labels

    def _read_paths_from_file(self, file_name):
        with open(file_name, 'r') as f:
            #return [line.strip().split(',')[0] for line in f]  # 只取路径部分
            return [line.strip() for line in f]  # 读取每行作为路径

    def __len__(self):
        return len(self.filtered_indices)

    def __getitem__(self, idx):
        actual_idx = self.filtered_indices[idx]  # Get actual index based on filter_interval
        inp = self.input_stack[actual_idx]
        tg = self.target_labels[actual_idx]
        # 直接使用 tg，不再尝试索引特定位置

        if inp.dim() == 2:
            inp = torch.unsqueeze(inp, 0)
        assert inp.dim() == 3

        if self.horizontal_flip:
            if random.random() > 0.5:
                inp = torch.flip(inp, [2])
        if self.vertical_flip:
            if random.random() > 0.5:
                inp = torch.flip(inp, [1])
        if self.rotate:
            if random.random() > 0.5:
                num_rotations = random.randint(0, 3)
                inp = torch.rot90(inp, num_rotations, [1, 2])
                if num_rotations in (1, 3):
                    # restore original shape
                    rot_shape = inp.shape
                    if rot_shape[2] > rot_shape[1]:
                        diff = (rot_shape[2] - rot_shape[1]) // 2
                        pad = (0, 0, diff + 1, diff + 1)
                        inp = torch.nn.functional.pad(inp, pad, "constant", 0)
                        inp = inp[:, 0:rot_shape[2], diff:diff + rot_shape[1]]
                    if rot_shape[1] > rot_shape[2]:
                        diff = (rot_shape[1] - rot_shape[2]) // 2
                        pad = (diff + 1, diff + 1, 0, 0)
                        inp = torch.nn.functional.pad(inp, pad, "constant", 0)
                        inp = inp[:, diff:diff + rot_shape[2], 0:rot_shape[1]]
        if self.random_crop:
            pad = (self.padding, self.padding, self.padding, self.padding)
            inp = torch.nn.functional.pad(inp, pad, "constant", 0)
            i = random.randint(0, 2 * self.padding)
            j = random.randint(0, 2 * self.padding)
            inp = inp[:, i:i + self.crop_shape[0], j:j + self.crop_shape[1]]

        return inp, tg

    def check_class_frequency(self):
        unique, counts = np.unique(self.target_labels, return_counts=True)
        print(unique)
        print(counts)
