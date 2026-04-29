import re

import numpy as np
from pathlib import Path

from matplotlib import pyplot as plt
from natsort import natsorted
from scipy import ndimage
from termcolor import colored
import flexsim

def plot_volume(volume, title="Volume", save_path=None):
    """
    可视化3D体积，突出显示异物
    volume: 3D ndarray
    """
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # 主体体素 (1)
    x, y, z = np.where(volume == 1)
    ax.scatter(x, y, z, zdir='z', c='gray', alpha=0.2, s=1, label='Main object')

    # 异物体素 (2)
    labels, nfeatures = ndimage.label(volume == 2)
    colors = ['red', 'blue', 'green', 'yellow']  # 不同异物用不同颜色
    for i in range(1, nfeatures + 1):
        xi, yi, zi = np.where(labels == i)
        color = colors[(i-1) % len(colors)]
        ax.scatter(xi, yi, zi, zdir='z', c=color, alpha=0.8, s=5, label=f'FO {i}')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    ax.legend()
    if save_path is not None:
        save_path = Path(save_path)
        if not save_path.exists():
            fig.savefig(save_path)
            print(f"Saved volume plot to: {save_path}")
        else:
            print(f"!!!!!Volume plot already exists, skipped saving: {save_path}")
    # plt.show()

def print_colored(text, color):
    print(colored(text, color))

def write_playdoh_volume_stats(obj, original_folder):  # 定义函数 write_playdoh_volume_stats，用于记录体积统计信息。
    mat_counts = obj.get_stats()  # mat_counts 保存 obj 对象中材料的统计信息。
    print("Material counts:", mat_counts)  # 添加调试信息

    labels, nfeatures = ndimage.label(obj.volume == 2)
    # 使用 ndimage.label 对体积中值为2的部分进行连通区域标记，返回 labels 和 nfeatures（连通区域的数量）。labels是一个标记后的数组，其中每个连通区域（值为2的部分）都被分配了一个唯一的整数标签
    sample_class = nfeatures  # 将连通区域的数量赋值给 sample_class
    stat_line = "{},{},{}".format(",".join(str(num) for num in mat_counts), nfeatures, sample_class)
    # 格式化统计信息，生成 stat_line 字符串
    with open(original_folder / "volume_info.csv", "w") as f:
        # 打开 out_folder1 中的 volume_info.csv 文件，准备写入，如果该文件不存在，Python 会自动创建这个文件
        f.write("Playdoh,Pebble,Pebble_objects,Sample_class\n")  # 写入CSV文件头部
        f.write(stat_line)  # 写入统计信息 stat_line



def playdoh_basic_augmentation(config_fname, input_root, input_folder, csv_folder, remove_fo=0):
    """ Create an artificial volume and the corresponding X-ray projections.不移除异物
    This function is used for replication and basic augmentation, it only allows FO modification.
    """
    config = flexsim.utils.read_config(config_fname)  # 读取配置文件 config_fname，内容存储在 config 中
    obj_vol_folder = input_folder / "segm"  # 设置对象体积文件夹路径 obj_vol_folder

    energy_bins = config['Simulation']['energy_bins']  # 从配置文件中读取能量区间 energy_bins

    obj_shape = flexsim.utils.get_volume_properties(obj_vol_folder)  # 获取对象体积的属性 obj_shape
    mat = flexsim.MaterialHandler(energy_bins, config['Materials'])  # 创建材料处理器 mat，使用能量区间和配置文件中的材料信息
    obj = flexsim.ObjectCreator(obj_shape, mat)  # 创建对象生成器 obj，使用体积属性和材料处理器
    obj.set_flexray_volume(obj_vol_folder)  # 设置 Flexray 体积

    folder_3d = input_root / "3D图"
    if not folder_3d.exists():
        folder_3d.mkdir(exist_ok=True)  # 不创建父目录
    plot_volume(obj.get_volume(),
                title=input_folder.name,  # 子文件夹名作为标题
                save_path=folder_3d / f"{input_folder.name}.png")
    if csv_folder.exists():
        print(f"{csv_folder} 已存在，跳过处理。")
    else:
        write_playdoh_volume_stats(obj, input_folder)  # 写入体积统计信息
        print("csv写入成功")

def batch_replication(input_root, config_fname):
    """ Simulate X-ray projections based on their reconstructions without any volume modification.
    This function is used to compare the neural network performance when trained on artifical data with training on real data.
    Noise properties can be changed in config to test noiseless and noisy data
    """
    subfolders = [path for path in input_root.iterdir() if path.is_dir()]#遍历这个迭代器，并且只选择那些是目录（文件夹）的路径，将这些路径存储在 subfolders 列表中。
    subfolders = natsorted([path.name for path in subfolders])
    print(subfolders)
    round_counter = 1
    pattern = re.compile(r'^Object\d+_Scan20W$')

    for i in range(len(subfolders)):  #遍历 subfolders 列表中的每个元素
        subfolder = subfolders[i]  # 通过索引获取当前子文件夹名称
        # 定义源文件夹和带 `_noisy` 后缀的文件夹路径
        if not pattern.fullmatch(subfolder):
            print(f"跳过（非 Object*_Scan20W）: {subfolder}")
            continue  # 跳过
        original_folder = input_root / subfolder

        csv_folder = original_folder / "volume_info.csv"  # 定义 volume_info.csv 文件的路径
        # 检查是否已经处理过，避免重复处理


        print(f'第{round_counter}轮: 处理 {original_folder}')
        round_counter += 1

        # 调用 playdoh_basic_augmentation 方法，生成有csv
        playdoh_basic_augmentation(config_fname, input_root,original_folder, csv_folder, 0)


if __name__ == "__main__":
    config_fname = "playdoh.ini"  # 设置配置文件名
    config = flexsim.utils.read_config('playdoh.ini')  # 读取配置文件
    input_root = Path(r"M:\m\pycharm_project\project1\Data\Real\Playdoh0-3\Playdoh2")

    np.random.seed(seed=6)  # 设置随机数种子

    batch_replication(input_root, config_fname)
    # 1 为所有分割对象生成带噪声的模拟投影，而不改变其内部结构。此功能可用于用模拟图像替换获取的投影，并检查它如何影响机器学习算法。
