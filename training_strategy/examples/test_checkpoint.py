import logging
import time

import torch
import argparse
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
import pandas as pd
import aug_accuracy as util

def setup_logger(log_path):
    logger2 = logging.getLogger('training_logger')
    logger2.setLevel(logging.INFO)
    file_handler = logging.FileHandler(Path(log_path) / f'{run_num}_train.log')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger2.addHandler(file_handler)
    # logger2.info("Logger setup complete.")
    return logger2

def create_file_paths(data_folder, phase):
    input_paths = []
    target_paths = []
    data_folder_path = Path(data_folder)

    if not data_folder_path.is_dir():
        raise ValueError(f"Data folder {data_folder} does not exist or is not a directory")
    subfolders = sorted([subfolder for subfolder in data_folder_path.iterdir() if subfolder.is_dir()])

    for subfolder in subfolders:
        log_folder = Path(subfolder) / 'log'
        tiff_paths = sorted(log_folder.glob('*.tiff'))
        input_paths.extend([str(p) for p in tiff_paths])

        volume_info_path = Path(subfolder) / 'volume_info.csv'
        if volume_info_path.exists():
            target_paths.extend([str(volume_info_path)] * len(tiff_paths))
        else:
            print(f"Warning: {volume_info_path} does not exist")

    return input_paths, target_paths

def create_path_file(path_list, file_name):
    with open(file_name, 'w') as f:
        for path in path_list:
            f.write(f"{path}\n")

def load_volume_info(volume_info_path):
    try:
        df = pd.read_csv(volume_info_path)
        df.columns = df.columns.str.strip()
        if 'Sample_class' not in df.columns:
            raise KeyError("The column 'Sample_class' is missing in the CSV file")
        return df['Sample_class'].values
    except Exception as e:
        print(f"Error loading volume_info.csv from {volume_info_path}: {e}")
        raise

def generate_target_labels(input_paths_file, target_paths_file, output_file):
    input_paths = []
    target_paths = []

    with open(input_paths_file, 'r') as infile:
        input_paths = [line.strip() for line in infile if line.strip()]

    with open(target_paths_file, 'r') as infile:
        target_paths = [line.strip() for line in infile if line.strip()]

    target_dict = {}
    for volume_info_path in set(target_paths):
        try:
            labels = load_volume_info(volume_info_path)
            target_dict[volume_info_path] = labels
        except Exception as e:
            print(f"Error processing {volume_info_path}: {e}")
            continue

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


def test(config, data_folder, epoch_file, data_type, nn_type):

    log_path = "../log/{}/".format(base_name)
    logger2 = setup_logger(log_path)

    start_time = time.time()
    start_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time))
    print(f"测试开始时间: {start_time_str}")
    print(f"\nStarting test with data_folder: {data_folder} \nepoch_file: {epoch_file}")

    data_folder = "{}/{}".format(test_folder, dataset_name)
    test_input_paths, test_target_paths = create_file_paths("{}/Test".format(data_folder), 'Testing')
    create_path_file(test_input_paths, 'test_input_paths.txt')
    create_path_file(test_target_paths, 'test_target_paths.txt')

    logger2.info("---------------------检查点，Testing started.---------------------")

    # 生成训练集目标标签
    test_target_file = 'test_target_labels.txt'
    generate_target_labels('test_input_paths.txt', 'test_target_paths.txt', test_target_file)


    batch_size = 1
    test_ds = util.ImageDatasetTransformable("test_input_paths.txt", 'test_target_labels.txt', config[data_type],
                                     random_crop=False, padding=20, crop_shape=(380, 478), vertical_flip=False,
                                     horizontal_flip=False, rotate=False, filter_interval=5)
    #测试切片下取样5，360/5=72
    print(f"处理总数: {len(test_ds)}")
    test_dl = DataLoader(test_ds, batch_size, shuffle=False)
    c_in = config[data_type]['c_in']
    c_out = config[data_type]['c_out']
    assert c_in == 1

    model = util.NNmodel(c_in, c_out, nn_type)
    print(f"Loading model from {epoch_file}")
    model.load(epoch_file)
    print("Starting model test")
    conf_mat, y_pred, y_true = model.test(test_dl)
    print("/////////Confusion matrix:")
    print(conf_mat)

    logger2.info("/////////Confusion matrix:")
    for row in conf_mat:
        logger2.info(" ".join(f"{int(x):6d}" for x in row))
    recalls = conf_mat.diagonal() / conf_mat.sum(axis=1)

    logger2.info("/////////Recall per class:")
    for i, r in enumerate(recalls):
        line = f"Class {i} Recall = {r:.6f}"
        logger2.info(line)
        print(line)
    macro_recall = recalls.mean()
    logger2.info(f"Macro Recall = {macro_recall:.6f}")
    print(f"Macro Recall = {macro_recall:.6f}")

    num_classes = config[data_type]['c_out']
    images_per_object = config[data_type]['img_per_obj']
    print("TP predictions split by objects:")
    print(util.utils.prediction_per_object(y_pred, y_true, num_classes, images_per_object))
    accuracy = util.utils.compute_accuracy(y_pred, y_true)
    print("Average accuracy = {:.5f}".format(accuracy))
    logger2.info("模型名：{}, 检查点迭代：{} , 准确率: {:.6f}".format(folder.name, best_epoch, accuracy))
    # logger2.info("---------------------Testing end.---------------------")
    total_time = time.time() - start_time
    # 将总时间转换为小时
    total_hours = total_time / 60
    # logger2.info(f"检查点Testing completed in {total_hours:.2f} 分钟")
    # logger2.info("---------------------Testing end.---------------------")
    generate_activation_maps = False
    # 改为True可以继续生成激活图
    if generate_activation_maps:
        cam = util.ActivationMap(model)
        # 测试模型并生成激活图
        model.classifier.eval()
        img_folder = Path("../image")
        img_folder.mkdir(exist_ok=True)
        i = 0
        for (inp, tg) in tqdm(test_dl):
            tg = tg.item()
            out = model._classify(inp)
            out_class = torch.max(out, 1).indices.item()
            if (i % 1 == 0) or (out_class != tg):
                cam.visualize(inp, tg, out_class, img_folder / "{}.png".format(i))
            i += 1
    # logger = util.Logger(log_path, model)
    # logger.check_validation(test_dl, 0)
    # # 记录测试准确率
    return accuracy

if __name__ == "__main__":
    config = util.utils.read_config('config.ini')
    data_keys = util.utils.get_available_data_types(config)
    parser = argparse.ArgumentParser()
    parser.add_argument('--nn', type=str, required=True, help='Network architecture')
    parser.add_argument('--data', type=str, required=True, help='Folder with the training set')
    parser.add_argument('--obj', type=str, required=True, choices=data_keys, help='Type of the dataset')
    parser.add_argument('--run', type=int, required=True, help='Run number')
    args = parser.parse_args()
    data_root = "../network_state"
    dataset_name = args.data
    data_type = args.obj
    nn_type = args.nn
    run_num = args.run
    test_folder = config['playdoh']['test_folder']
    data_root = Path(data_root)

    test_file = config['General']['test_file3']
    base_name = "{}".format(test_file)

    print("Base name:", base_name)
    subfolders = [x for x in data_root.iterdir() if x.is_dir()]
    subfolders = filter(lambda x: x.name.startswith(base_name), subfolders)
    subfolders = sorted(subfolders)
    run_epochs = []
    acc_values = []

    use_only_best = False
    # use_only_best = True

    for folder in tqdm(subfolders):
        saves = [x.stem for x in folder.glob("*.torch")]
        checkpoint_num = config['General']['test_checkpoint_num']
        best_checkpoint = f"checkpoint_{checkpoint_num}"
        # best_checkpoint = f"epoch_{epoch_num}"

        # 如果指定了特定的 checkpoint，则直接加载该文件

        best_epoch = best_checkpoint.split('_')[-1]

        epoch_file = folder / "{}.torch".format(best_checkpoint)

        accuracy = test(config, folder, epoch_file, data_type, nn_type)

        print("模型名：{}, 检查点迭代：{}".format(folder.name, best_epoch))


