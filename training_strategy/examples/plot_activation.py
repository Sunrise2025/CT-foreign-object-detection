import torch
import argparse
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
import pandas as pd
import aug_accuracy as util

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


def test(config, data_folder, epoch_file, data_type, nn_type, filter_interval=None):
    print(f"Starting test with data_folder: {data_folder}, epoch_file: {epoch_file}")
    data_folder = "{}/{}".format(test_folder, dataset_name)
    test_input_paths, test_target_paths = create_file_paths("{}/Test".format(data_folder), 'Testing')
    create_path_file(test_input_paths, 'test_input_paths.txt')
    create_path_file(test_target_paths, 'test_target_paths.txt')

    # 生成训练集目标标签
    test_target_file = 'test_target_labels.txt'
    generate_target_labels('test_input_paths.txt', 'test_target_paths.txt', test_target_file)

    log_path = "../log/{}_{}_r{}/".format(dataset_name, nn_type, run_num)

    batch_size = 1
    test_ds = util.ImageDatasetTransformable("test_input_paths.txt", 'test_target_labels.txt', config[data_type],
                                     random_crop=False, padding=20, crop_shape=(380, 478), vertical_flip=False,
                                     horizontal_flip=False, rotate=False, filter_interval=5)
    print(f"Using filter interval of {filter_interval}. Total number of images processed: {len(test_ds)}")
    test_dl = DataLoader(test_ds, batch_size, shuffle=False)
    c_in = config[data_type]['c_in']
    c_out = config[data_type]['c_out']
    assert c_in == 1

    model = util.NNmodel(c_in, c_out, nn_type)
    print(f"Loading model from {epoch_file}")

    model.load(epoch_file)
    print("Starting model test")
    conf_mat, y_pred, y_true = model.test(test_dl)
    print("Confusion matrix:")
    print(conf_mat)
    num_classes = config[data_type]['c_out']
    images_per_object = config[data_type]['img_per_obj']
    print("TP predictions split by objects:")
    print(util.utils.prediction_per_object(y_pred, y_true, num_classes, images_per_object))
    accuracy = util.utils.compute_accuracy(y_pred, y_true)
    print("Average accuracy = {:.3f}".format(accuracy))

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
    # # 返回验证准确率

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

    use_only_best = True

    data_root = Path(data_root)

    base_name = "{}_{}_r2".format("gen_train008", nn_type)
 #
    print("Base name:", base_name)

    subfolders = [x for x in data_root.iterdir() if x.is_dir()]
    subfolders = filter(lambda x: x.name.startswith(base_name), subfolders)
    subfolders = sorted(subfolders)

    run_epochs = []
    acc_values = []

    for folder in tqdm(subfolders):
        saves = [x.stem for x in folder.glob("*.torch")]

        if use_only_best:
            only_final = filter(lambda x: not x.startswith('checkpoint'), saves)
            epochs = [int(x) for x in only_final]
            best_epoch = max(epochs)
            epoch_file = folder / "{}.torch".format(best_epoch)
        else:
            saves = sorted(saves)
            best_checkpoint = saves[-1]
            best_epoch = best_checkpoint.split('_')[-1]
            epoch_file = folder / "{}.torch".format(best_checkpoint)
        print("Network {}, epoch {}".format(folder.name, best_epoch))
        accuracy = test(config, folder, epoch_file, data_type, nn_type)
        run_epochs.append(best_epoch)
        acc_values.append(accuracy)

    pairs = []
    for i in range(len(run_epochs)):
        pairs.append("{},{:.3f}".format(run_epochs[i], acc_values[i]))
    print(",".join(pairs))


