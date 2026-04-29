import numpy as np
import torch
import random
import argparse
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
import aug_accuracy as util
import pandas as pd
import logging
import time
import csv
import os

class Logger:
    def __init__(self, log_dir, model):
        self.log_dir = log_dir
        self.model = model
        self.train_log_file = os.path.join(log_dir, 'train_log.csv')
        self.val_log_file = os.path.join(log_dir, 'val_log.csv')
        self.setup_log_files()

    def setup_log_files(self):
        # Create log files and write header
        with open(self.train_log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'train_loss', 'train_accuracy'])

        with open(self.val_log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'val_loss', 'val_accuracy'])

    def log_train(self, loss, accuracy, epoch):
        # Log training loss and accuracy
        with open(self.train_log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, loss, accuracy])

    def log_validation(self, loss, accuracy, epoch):
        # Log validation loss and accuracy
        with open(self.val_log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, loss, accuracy])


def create_file_paths(data_folder, phase):
    input_paths = []
    target_paths = []
    data_folder_path = Path(data_folder)
    if not data_folder_path.is_dir():
        raise ValueError(f"Data folder {data_folder} does not exist or is not a directory")
    subfolders = sorted([subfolder for subfolder in data_folder_path.iterdir() if subfolder.is_dir()])

    for subfolder in subfolders:
        # 处理 .tiff 文件路径
        log_folder = Path(subfolder) / 'log'
        tiff_paths = sorted(log_folder.glob('*.tiff'))
        input_paths.extend([str(p) for p in tiff_paths])  # 收集所有TIFF文件的路径，并将它们以字符串的形式添加到input_paths列表中

        # 处理 volume_info.csv 文件路径
        volume_info_path = Path(subfolder) / 'volume_info.csv'
        if volume_info_path.exists():
            # print(f"Volume info path: {volume_info_path}")  # 调试输出
            target_paths.extend([str(volume_info_path)] * len(tiff_paths))
        else:
            print(f"Warning: {volume_info_path} does not exist")
            # 将 input_paths 和 target_paths 中的路径转换为 data_folder 同级路径

    return input_paths, target_paths


def create_path_file(path_list, file_name):
    with open(file_name, 'w') as f:
        for path in path_list:
            f.write(f"{path}\n")


def generate_target_labels(input_paths_file, target_paths_file, output_file):
    """
    从 volume_info.csv 文件生成目标标签。
    origin_train_input_paths.txt,   origin_train_target_paths.txt,   origin_train_target_labels.txt
    """
    input_paths = []
    target_labels = []

    # 读取所有输入路径
    with open(input_paths_file, 'r') as infile:  # origin_train_input_paths.txt
        input_paths = [line.strip() for line in infile if line.strip()]
        # 列表将包含origin_train_input_paths.txt文件中所有非空行的内容，每行内容作为一个独立的字符串元素

        # 读取目标路径文件
    with open(target_paths_file, 'r') as infile:  # origin_train_target_paths.txt
        target_paths = [line.strip() for line in infile if line.strip()]
        # 列表将包含origin_train_target_paths.txt文件中所有非空行的内容，每行内容作为一个独立的字符串元素

        # 读取每个 volume_info.csv 文件，提取标签
    target_dict = {}
    for volume_info_path in set(target_paths):  # 处理每个目标路径
        try:
            labels = load_volume_info(volume_info_path)  # 返回Sample_class列中的所有值

            target_dict[volume_info_path] = labels
        except Exception as e:
            print(f"Error processing {volume_info_path}: {e}")
            continue

    # 创建目标标签文件
    with open(output_file, 'w') as outfile:
        for input_path in input_paths:
            parent_dir = Path(input_path).parents[1]  # 获取父目录
            volume_info_path = next((p for p in target_paths if Path(p).parent == parent_dir), None)
            if volume_info_path:
                labels = target_dict.get(volume_info_path, [])
                index = input_paths.index(input_path) % len(labels) if len(labels) > 0 else None
                if index is not None:
                    outfile.write(f"{input_path},{labels[index]}\n")
                else:
                    print(f"Label not found for {input_path}")
            else:
                print(f"No volume_info.csv found for {input_path}")


def load_volume_info(volume_info_path):
    # 读取 volume_info.csv 文件内容
    try:
        df = pd.read_csv(volume_info_path)
        df.columns = df.columns.str.strip()  # 去除列名中的前后空格
        # print("Columns:", df.columns.tolist())  # 打印列名列表用于调试
        # print("Shape:", df.shape)  # 打印 DataFrame 形状用于调试
        # print(df.head())  # 打印前几行数据用于调试
        # 确保 'Sample_class' 列存在并且没有空格

        if 'Sample_class' not in df.columns:
            raise KeyError("The column 'Sample_class' is missing in the CSV file")
        return df['Sample_class'].values
    except Exception as e:
        print(f"Error loading volume_info.csv from {volume_info_path}: {e}")
        raise


def setup_logger(log_path):
    logger1 = logging.getLogger('training_logger')
    logger1.setLevel(logging.INFO)
    file_handler = logging.FileHandler(Path(log_path) / 'training_log.log')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger1.addHandler(file_handler)
    logger1.info("Logger setup complete.")
    return logger1


def train(config, dataset_name, data_type, nn_type, run_num, validation_metrics=None, filter_interval=None):
    # 读取配置文件中的根数据目录data_root = /training
    log_path = "../log/{}_{}_r{}/".format(dataset_name, nn_type, run_num)
    Path(log_path).mkdir(parents=True, exist_ok=True)
    logger = setup_logger(log_path)
    # 记录训练开始的时间
    start_time = time.time()
    logger.info("Training started.")
    # 检查 GPU 是否可用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # 打印 GPU 信息
    if torch.cuda.is_available():
        print(f"使用 GPU: {torch.cuda.get_device_name(device)}")
        print(f"Number of GPUs available: {torch.cuda.device_count()}")
    else:
        print("没有 GPU available, 使用 CPU.")

    data_folder = "{}/{}".format(data_root, dataset_name)
    # 根据数据集名称构建数据文件夹路径，dataset_name 是 data_root 目录下的一个子目录的名称,这里是Real/Playdoh0-3/gen_train
    origin_train_input_paths, origin_train_target_paths = create_file_paths("{}/training".format(data_folder), 'training')
    origin_val_input_paths, origin_val_target_paths = create_file_paths("{}/validation".format(data_folder), 'validation')
    # print(f"origin_train_input_paths: {origin_train_input_paths}")
    # 创建路径文件
    create_path_file(origin_train_input_paths, 'origin_train_input_paths.txt')
    create_path_file(origin_train_target_paths, 'origin_train_target_paths.txt')
    create_path_file(origin_val_input_paths, 'origin_val_input_paths.txt')
    create_path_file(origin_val_target_paths, 'origin_val_target_paths.txt')

    # 生成训练集目标标签
    origin_train_target_file = 'origin_train_target_labels.txt'
    generate_target_labels('origin_train_input_paths.txt', 'origin_train_target_paths.txt', origin_train_target_file)

    # 生成验证集目标标签
    origin_val_target_file = 'origin_val_target_labels.txt'
    generate_target_labels('origin_val_input_paths.txt', 'origin_val_target_paths.txt', origin_val_target_file)

    save_path = Path("../network_state") / f"{dataset_name}_{nn_type}_r{run_num}"
    log_path = "../log/{}_{}_r{}/".format(dataset_name, nn_type, run_num)
    # 构建保存模型和日志的路径
    Path(save_path).mkdir(parents=True, exist_ok=True)
    Path(log_path).mkdir(parents=True, exist_ok=True)
    # 确保保存路径和日志路径存在，如果不存在则创建

    batch_size = config['General']['batch_size']
    # 根据配置文件中的设置确定批处理大小
    print('开始train')
    train_ds = util.ImageDatasetTransformable('origin_train_input_paths.txt', 'origin_train_target_labels.txt', config[data_type],
                                              random_crop=True, padding=20, crop_shape=(380, 478), vertical_flip=True,
                                              horizontal_flip=True, rotate=True, filter_interval=5)
    print(f"训练Using filter interval of {filter_interval}. Total number of images processed: {len(train_ds)}")
    # 创建可变形的训练数据集，util.ImageDatasetTransformable类（或类似功能的类）通常用于深度学习和计算机视觉项目中，以增加模型的泛化能力，避免过拟合，并提高模型的鲁棒性。
    train_dl = DataLoader(train_ds, batch_size, shuffle=True)  # 创建训练数据加载器

    val_ds = util.ImageDatasetTransformable('origin_val_input_paths.txt', 'origin_val_target_labels.txt', config[data_type],
                                            random_crop=False, padding=20, crop_shape=(380, 478), vertical_flip=False,
                                            horizontal_flip=False, rotate=False, filter_interval=5)  # 创建可变形的验证数据集 # 验证数据加载
    val_dl = DataLoader(val_ds, batch_size, shuffle=False)
    # 创建验证数据加载器
    print(f"验证Using filter interval of {filter_interval}. Total number of images processed: {len(val_ds)}")

    train_ds.check_class_frequency()
    val_ds.check_class_frequency()
    # 检查训练集和验证集中的类别频率

    c_in = config[data_type]['c_in']
    c_out = config[data_type]['c_out']
    # 从配置文件中提取输入通道数和输出通道数

    model = util.NNmodel(c_in, c_out, nn_type)  # 初始化神经网络模型
    model.to(device)  # 将模型转移到 GPU
    model.set_normalization(train_dl)  # 根据训练数据设置归一化参数

    best_validation_loss = np.inf
    prev_best_epoch = -1
    only_best_torch = True
    # 初始化最佳验证损失为无穷大
    logger = util.Logger(log_path, model)
    # 初始化日志记录器

    # Edit the number of epochs depending on the convergence
    epochs = config['General']['max_epochs']
    # 从配置文件中获取最大训练轮次数

    for epoch in tqdm(range(epochs)):  # 迭代训练循环，使用tqdm显示进度条
        epoch_start_time = time.time()
        train_loss = model._train(train_dl)  # 训练模型并获取训练损失# 传递 device 参数
        epoch_end_time = time.time()
        logger.log_train(train_loss, epoch)  # 记录训练损失到日志
        validation_loss = model._validate(val_dl)  # 验证模型并获取验证损失
        epoch_duration = epoch_end_time - epoch_start_time
        logger.info(f"Epoch {epoch}: Train loss: {train_loss}")
        logger.info(f"Epoch {epoch}: Time taken for this epoch: {epoch_duration:.2f} seconds")
        logger.log_validation(validation_loss, epoch)  # 记录验证损失到日志
        logger.info(f"Epoch {epoch}: Validation loss: {validation_loss}")

        # 每隔50轮检查一次验证损失
        if epoch % 25 == 0:
            logger.info(f"Epoch {epoch}: Checking validation performance")
            logger.accuracy_validation(val_dl, epoch)

        if validation_loss < best_validation_loss:  # 如果当前验证损失优于历史最佳验证损失，则更新最佳模型
            best_validation_loss = validation_loss  # 保存当前最佳模型
            model.save(save_path / "{}.torch".format(epoch), epoch, batch_size)
            if only_best_torch == True:
                if prev_best_epoch != -1:
                    os.remove(save_path / "{}.torch".format(prev_best_epoch))  # 删除之前保存的历史最佳模型
                prev_best_epoch = epoch

        if epoch % 100 == 0:
            model.save(save_path / "checkpoint_{}.torch".format(epoch), epoch, batch_size)
            logger.info(f"Checkpoint saved for epoch {epoch}")
        # 每隔100轮保存一次检查点模型
    total_time = time.time() - start_time
    logger.info(f"Training completed in {total_time:.2f} seconds")


if __name__ == "__main__":
    config = util.utils.read_config('config.ini')  # 读取配置文件
    data_keys = util.utils.get_available_data_types(config)  # 获取可用的数据类型列表

    # 设置命令行参数解析器
    parser = argparse.ArgumentParser()
    parser.add_argument('--nn', type=str, required=True, help='Network architecture')
    # 一个字符串类型的参数，必须提供，用于指定网络架构
    parser.add_argument('--data', type=str, required=True, help='Folder with the training set')
    # 字符串类型的参数，必须提供，用于指定包含训练集的文件夹
    parser.add_argument('--obj', type=str, required=True, choices=data_keys, help='Type of the dataset')
    # 字符串类型，必须提供，但还限制了可选值（通过choices参数），这通常是从一个预定义的列表data_keys中选取，用于指定数据集的类型。
    parser.add_argument('--run', type=int, required=True, help='Run number')
    # 整数类型的参数，也必须提供，用于指定运行的编号。
    parser.add_argument('--seed', type=int, required=False,
                        help='Random seed. Optional, run number will be used as a seed if this argument is not provided')
    # 可选的整数类型参数，用于指定随机种子。如果未提供，根据注释，运行编号将被用作随机种子。
    args = parser.parse_args()
    # 获取数据根目录和命令行参数的值,从config字典中获取数据根目录，并从args对象中提取命令行参数的值
    data_root = config['General']['data_root']
    dataset_name = args.data
    data_type = args.obj
    nn_type = args.nn
    run_num = args.run
    if args.seed is not None:  # 根据参数或运行编号确定随机种子
        random_seed = args.seed
    else:
        random_seed = run_num

    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    # 设置随机种子以保证实验的可重现性
    if config['General']['use_deterministic']:  # 根据配置文件中的设置决定是否使用确定性算法
        print("Use detereterministic algorithms")
        torch.use_deterministic_algorithms(True)  # 设置PyTorch使用确定性算法
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # 设置CUDA BLAS工作空间配置以确保确定性行为
    else:
        print("Use faster algorithms")
    # 开始训练过程
    train(config, dataset_name, data_type, nn_type, run_num)
