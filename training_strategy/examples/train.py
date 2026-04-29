import shutil

import numpy as np
import torch
import random
import argparse
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
from examples import aug_accuracy as util
import pandas as pd
import logging
import time
import os


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
    train_input_paths.txt,   train_target_paths.txt,   train_target_labels.txt
    """

    # 读取所有输入路径
    with open(input_paths_file, 'r') as infile:  # train_input_paths.txt
        input_paths = [line.strip() for line in infile if line.strip()]
        # 列表将包含train_input_paths.txt文件中所有非空行的内容，每行内容作为一个独立的字符串元素

        # 读取目标路径文件
    with open(target_paths_file, 'r') as infile:  # train_target_paths.txt
        target_paths = [line.strip() for line in infile if line.strip()]
        # 列表将包含train_target_paths.txt文件中所有非空行的内容，每行内容作为一个独立的字符串元素

        # 读取每个 volume_info.csv 文件，提取标签
    target_dict = {}
    for volume_info_path in set(target_paths):  # 处理每个目标路径
        try:
            labels = load_volume_info(volume_info_path)  # 返回Sample_class列中的所有值
            # label_count = len(labels)#label_count变量被赋值为labels列表的长度
            # matching_paths = [p for p in input_paths if Path(p).parent == Path(volume_info_path).parent]
            #  if len(matching_paths) != label_count:
            #      raise ValueError(
            #          f"Label count mismatch for {volume_info_path}. Expected {len(matching_paths)}, but got {label_count}")
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
    logger2 = logging.getLogger('training_logger')
    logger2.setLevel(logging.INFO)
    file_handler = logging.FileHandler(Path(log_path) / f'{run_num}_train.log')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger2.addHandler(file_handler)
    return logger2


def train(config, dataset_name, data_type, nn_type, run_num):
    # 读取配置文件中的根数据目录data_root = /training
    log_path = "../log/{}_{}_r{}/".format(dataset_name, nn_type, run_num)
    Path(log_path).mkdir(parents=True, exist_ok=True)
    # 记录训练开始的时间
    logger2 = setup_logger(log_path)
    logger2.info("-------------------------------------Training started.-------------------------------------")

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
    # dataset_name 是 data_root = Data/Real/Playdoh0-3/gen_train目录下的一个子目录的名称；
    train_input_paths, train_target_paths = create_file_paths("{}/training".format(data_folder), 'training')
    val_input_paths, val_target_paths = create_file_paths("{}/validation".format(data_folder), 'validation')
    # print(f"train_input_paths: {train_input_paths}")

    # 将 txt 文件放在 log_path 下
    train_input_txt = Path(log_path) / "train_input_paths.txt"
    train_target_txt = Path(log_path) / "train_target_paths.txt"
    val_input_txt = Path(log_path) / "val_input_paths.txt"
    val_target_txt = Path(log_path) / "val_target_paths.txt"

    create_path_file(train_input_paths, train_input_txt)
    create_path_file(train_target_paths, train_target_txt)
    create_path_file(val_input_paths, val_input_txt)
    create_path_file(val_target_paths, val_target_txt)

    # 生成训练集目标标签
    train_target_file = Path(log_path) / "train_target_labels.txt"
    generate_target_labels(train_input_txt, train_target_txt, train_target_file)

    val_target_file = Path(log_path) / "val_target_labels.txt"
    generate_target_labels(val_input_txt, val_target_txt, val_target_file)

    save_path = Path("../network_state") / f"{dataset_name}_{nn_type}_r{run_num}"
    # log_path = "../log/{}_{}_r{}/".format(dataset_name, nn_type, run_num)
    # 构建保存模型和日志的路径
    Path(save_path).mkdir(parents=True, exist_ok=True)
    # 确保保存路径和日志路径存在，如果不存在则创建

    batch_size = config['General']['batch_size']
    # 根据配置文件中的设置确定批处理大小
    print('开始train')
    train_ds = util.ImageDatasetTransformable(str(train_input_txt), str(train_target_file),  config[data_type],
                                              random_crop=True, padding=20, crop_shape=(380, 478), vertical_flip=True,
                                              horizontal_flip=True, rotate=True)
    # 创建可变形的训练数据集，util.ImageDatasetTransformable类（或类似功能的类）通常用于深度学习和计算机视觉项目中，以增加模型的泛化能力，避免过拟合，并提高模型的鲁棒性。
    train_dl = DataLoader(train_ds, batch_size, shuffle=True)  # 创建训练数据加载器

    val_ds = util.ImageDatasetTransformable(str(val_input_txt), str(val_target_file),config[data_type],
                                            random_crop=False, padding=20, crop_shape=(380, 478), vertical_flip=False,
                                            horizontal_flip=False, rotate=False)  # 创建可变形的验证数据集 # 验证数据加载
    val_dl = DataLoader(val_ds, batch_size, shuffle=False)
    # 创建验证数据加载器
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

    best_window_score = float('inf')
    best_window_epochs = None  # (start, end)
    best_epoch = None
    best_epoch_saved = False
    temp_window_files = []  # 当前窗口的文件
    best_window_files = []  # 历史最佳窗口文件

    model_name = f"模型代号：{dataset_name}_{nn_type}_r{run_num}"
    logger = util.Logger(log_path, model, model_name)
    # 初始化日志记录器
    train_losses, val_losses =  [], []
    epochs = config['General']['max_epochs']
    # 从配置文件中获取最大训练轮次数
    recent_avg_val_max = float(config['General']['recent_avg_val_max'])

    improvement_values = []  # 记录每个 epoch 的改进差值
    patience =config['General']['patience']
    patience = int(patience)  # 将 patience 转换为整数类型
    previous_mean_diff = float('inf')
    min_improvement_threshold = config['General']['min_improvement_threshold']
    min_improvement_threshold = float(min_improvement_threshold)
    max_improvement_threshold = config['General']['max_improvement_threshold']
    max_improvement_threshold = float(max_improvement_threshold)
    early_stop_ratio = float(config['General']['early_stop_ratio'])

    improvement_flags = []    # 记录每个epoch的改进情况，True表示有改进，False表示没有改进
    count=0

    for epoch in tqdm(range(1, epochs + 1), position=1):  # 迭代从1到epochs (包括500)
        train_loss = model._train(train_dl)  # 训练模型并获取训练损失# 传递 device 参数
        logger.log_train(train_loss, epoch)  # 记录训练损失到日志
        # 在训练和验证损失计算之后，添加损失值到相应的列表
        train_losses.append(train_loss)  # 保存训练损失
        validation_loss = model._validate(val_dl)  # 验证模型并获取验证损失
        logger.log_validation(validation_loss, epoch)  # 记录验证损失到日志
        # 在训练和验证损失计算之后，添加损失值到相应的列表
        val_losses.append(validation_loss)  # 保存验证损失
        # 计算当前及之前的 25 个连续 epochs 损失差值绝对平均值

    # 早停条件检查
        if epoch >=50:
            val_slice = val_losses[epoch - 25: epoch]  # 获取验证损失的最后 25 个 epochs，epoch 是结束索引（不包含该位置）
            train_slice = train_losses[epoch - 25: epoch]  # 获取训练损失的最后 25 个 epochs
            avg_val_loss_25 = float(np.mean(val_slice))
            mean_diff = np.mean(np.abs(np.array(val_slice) - np.array(train_slice)))
            # 如果有任何一轮的平均差值小于最优标准，说明有改进
            improvement_value = previous_mean_diff - mean_diff  # 改进的差值
            count += 1
            if improvement_value > min_improvement_threshold:
                improvement_flag = True  # 有改进
                # no_improvement_count = 0  # 有改进，重置未改进计数器
                logger2.info(
                    f"改进区间，序号{count}，epoch：({epoch - 24}:{epoch}) | "
                    f"avg_val_loss_25 = {avg_val_loss_25:.6f} | "
                    f"mean_diff = {mean_diff:.6f} | "
                    f"Δdiff = {improvement_value:.6f}"
                )

            else:
                improvement_flag = False  # 没有改进
                logger2.info(
                    f"-----------未改进区间，序号{count}，epoch：({epoch - 24}:{epoch}) | "
                    f"avg_val_loss_25 = {avg_val_loss_25:.6f} | "
                    f"mean_diff = {mean_diff:.6f} | "
                    f"Δdiff = {improvement_value:.6f}"
                )


                # 将改进标志保存
            improvement_flags.append(improvement_flag)
            improvement_values.append(improvement_value)
            previous_mean_diff = mean_diff  # 更新前一段的均值差
            # 判断是否触发早停条件：在最近patience个epoch中，未改进次数超过90%的patience
            recent_no_improvement_count = sum(1 for flag in improvement_flags[-patience:] if not flag)
            #列表中最近 patience 个值为 False（表示没有改进）的元素个数。
            recent_improvement_values = improvement_values[-patience:]  # 获取最近 `patience` 个 epoch 的改进差值
            recent_avg_val = np.mean(val_losses[-patience:])  # 最近 patience 个 epoch 的平均验证损失

            # 如果在 `patience` 内，未改进的次数超过 early_stop_ratio%，则触发早停
            if recent_no_improvement_count >= int(early_stop_ratio * patience)  :
                # 检查最近 `patience` 个 epoch 中是否存在改进值超出阈值的情况
                if all(value < max_improvement_threshold for value in recent_improvement_values ) or  recent_avg_val > recent_avg_val_max:
                    final_epoch = epoch  # 记录最终 epoch
                    logger2.info(f"------------------------------------------------------------------------早停  {final_epoch} epoch ----------------------")
                    print(f"---------------------------------------早停 {final_epoch} epoch")
                    logger2.info(
                        f"batch_size={batch_size}，25epochs早停改进最小均值：{min_improvement_threshold}，改进最大均值：{max_improvement_threshold}，忍耐值：{patience} ")
                    break  # 触发早停
                # else:
                #     # logger2.info(f"在最近 {patience} 个 epochs 内，某些 epochs 的改进差值超过了最大阈值{max_improvement_threshold}，继续训练。")
                #     continue  # 如果不满足早停条件，则继续训练
            else:
                # logger2.info(f"未早停， epochs： {final_epoch}. Mean difference: {previous_mean_diff:.6f}")
                print(f"      Mean difference = {previous_mean_diff:.6f}")

        # 每隔25轮检查一次验证损失
        if epoch % 25 == 0:
            logger.check_validation(val_dl, epoch)
        # 返回验证准确率
        if validation_loss < best_validation_loss:  # 如果当前验证损失优于历史最佳验证损失，则更新最佳模型
            best_validation_loss = validation_loss  # 保存当前最佳模型
            model.save(save_path / "{}.torch".format(epoch), epoch, batch_size)
            if only_best_torch:
                if prev_best_epoch != -1:
                    os.remove(save_path / "{}.torch".format(prev_best_epoch))  # 删除之前保存的历史最佳模型
                prev_best_epoch = epoch
        if epoch % 100 == 0:
            model.save(save_path / "checkpoint_{}.torch".format(epoch), epoch, batch_size)
        # 每隔100轮保存一次检查点模型



#选取最优模型
        # 保存当前 epoch 模型
        curr_path = save_path / f"epoch_{epoch}.torch"
        model.save(curr_path, epoch, batch_size)
        # logger2.info(f"保存epoch_{epoch}模型文件")
        c = float(config['General']['select_weight'].split(',')[0])
        # ======================================================
        #   从第 3 个 epoch 开始，形成窗口 {epoch-2, epoch-1, epoch}
        # ======================================================
        if epoch >= 3:
            window_start = epoch - 2
            window_mid = epoch - 1
            window_end = epoch
            # logger2.info(f"形成窗口 {window_start}-{window_end}")
            # 当前窗口需要保留（只有后两个）
            current_keep = {window_mid, window_end}
            # --- 当前窗口的 losses（必须传入 3 个 epoch 的值） ---
            curr_window_val = val_losses[window_start - 1: window_end]
            # average_diff = np.mean(np.abs(np.array(curr_window_val) - np.array(curr_window_train)))  # 平均损失差的绝对值
            avg_val_loss = np.mean(curr_window_val)  # 平均验证损失
            # --- 使用你的 select_best_model() 得到 score ---
            curr_score = avg_val_loss
            avg_val_loss = np.mean(curr_window_val)
            # average_diff = np.mean(np.abs(np.array(curr_window_train) - np.array(curr_window_val)))
            # min_val_idx = int(np.argmin(curr_window_val))  # 0/1/2
            # best_epoch = window_start + min_val_idx

            if best_window_epochs is not None:
                logger2.info(
                    f"-----------当前窗口 {window_start}-{window_end} ，avg_val_loss= {avg_val_loss:.6f} , score= {curr_score:.6f} | "
                    f"目前最佳窗口 {best_window_epochs[0]}-{best_window_epochs[1]}，score= {best_window_score:.6f}"
                    # f" | 最佳 epoch = {best_epoch}, val_loss = {curr_window_val[min_val_idx]:.6f}"
                )
            else:
                # 如果还没有最佳窗口
                logger2.info(
                    f"-----------当前窗口 {window_start}-{window_end} ，avg_val_loss= {avg_val_loss:.6f} , score = {curr_score:.6f} "
                    f"目前没有最佳窗口"
                )
            # 计算平均验证损失和平均差值，用于新的条件

            # ====================================================================
            # 比较窗口 score，若当前窗口更优，则更新最佳窗口
            if curr_score <= best_window_score and avg_val_loss < c :
                logger2.info(f"//////////////窗口 {window_start}-{window_end} 是新的潜在最佳窗口，avg_val_loss= {avg_val_loss:.6f} , score= {curr_score:.6f}")
                # ------------------------------
                # 删除之前最佳窗口的模型文件
                # 更新最佳窗口信息
                best_window_score = curr_score
                best_window_epochs = (window_start, window_end)
                # ------------------------------
                # 记录该窗口中的 best_epoch
                # ------------------------------
                # 更新保留集合（包含当前窗口后两个 + 新的最佳窗口所有 epoch）
                b_start, b_end = best_window_epochs
                best_keep = {b_start, b_start + 1, b_end}
                keep_epochs = current_keep | best_keep

                min_val_idx = int(np.argmin(curr_window_val))  # 0/1/2
                best_epoch = window_start + min_val_idx
                logger2.info(f"////////////当前最佳窗口最佳 epoch = {best_epoch},  val_loss = {curr_window_val[min_val_idx]:.6f}")

            else:
                # 没有新的最佳窗口，仍然保留历史最佳 + 当前窗口后两个
                if best_window_epochs is not None:
                    b_start, b_end = best_window_epochs
                    best_keep = {b_start, b_start + 1, b_end}
                else:
                    best_keep = set()
                keep_epochs = current_keep | best_keep

            def safe_delete_all(save_path, keep_epochs, logger2):
                keep_epochs = set(keep_epochs)  # 保证是 set
                for f in save_path.glob("epoch_*.torch"):
                    try:
                        epoch_num = int(f.stem.split("_")[1])
                    except:
                        continue

                    if epoch_num not in keep_epochs:
                        f.unlink()
                        # logger2.info(f"安全删除 epoch 文件：{f.name}")
            safe_delete_all(save_path, keep_epochs, logger2)

    logger2.info(f"/////////原策略：{prev_best_epoch} epoch, val_loss = {best_validation_loss:.6f}")
    print(f"***原策略，Best epoch: {prev_best_epoch}, val_loss = {best_validation_loss:.6f}")

    # 显式引用变量以避免警告
    # ===================================================
    # 训练结束后，复制并重命名最终 best_epoch 模型
    # ===================================================
    if best_window_epochs is not None and best_epoch is not None:
        start_3, end_3 = best_window_epochs
        best_window_vals = val_losses[start_3 - 1: end_3]  # 3 个值
        min_val_loss = float(np.min(best_window_vals))
        avg_val_loss_3 = float(np.mean(best_window_vals))
        best_model_file = save_path / f"epoch_{best_epoch}.torch"
        final_name = save_path / f"select_best_{best_epoch}.torch"

        shutil.copy(best_model_file, final_name)
        logger2.info(
            f"/////////新策略：{best_epoch} epoch, "
            f"val_loss = {min_val_loss:.6f}, "
            f"最佳 3epochs = ({start_3}:{end_3}), "
            f"平均验证损失 = {avg_val_loss_3:.6f}"
        )
        print(f"***新策略，Best epoch: {best_epoch}, val_loss = {min_val_loss:.6f}")

        # a, b = map(float, config['General']['select_weight'].split(','))
        c = float(config['General']['select_weight'].split(',')[0])
        logger2.info(f"最优模型确定的权重值  c = {c}")

    else:
        logger2.info("未找到最佳 epoch，可能训练损失未满足条件")
        print(f"***新策略未找到最佳 epoch")

    total_time = time.time() - start_time
    # 将总时间转换为小时
    total_hours = total_time / 3600
    logger2.info(f"-------------------------------------Training end. 模型代号：{dataset_name}_{nn_type}_r{run_num}，用时 {total_hours:.2f} 小时--------------------")
    # logger2.info(f"模型代号：{dataset_name}_{nn_type}_r{run_num}，训练次数：{epochs}")

if __name__ == "__main__":

    start_time = time.time()
    start_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time))
    print(f"【开始时间】: {start_time_str}")
    config = util.utils.read_config('config.ini')  # 使用默认的 utf-8 编码
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
    # 可选的整数类型参数，用于指定随机种子。如果未提供，根 据注释，运行编号将被用作随机种子。
    args = parser.parse_args()
    # 获取数据根目录和命令行参数的值,从config字典中获取数据根目录，并从args对象中提取命令行参数的值
    data_root = config['General']['data_root']
    dataset_name = args.data
    #data这里是gen_train002

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
