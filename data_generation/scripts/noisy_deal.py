import os
import shutil
import pandas as pd

# 定义源目录和目标目录
source_dir = r"M:\m\pycharm_project\project1\Data\Real\Playdoh0-3\Playdoh0-3_segm"
destination_dir = r"M:\m\pycharm_project\project1\Data\Real\Playdoh0-3\Playdoh0-3_segm_加噪音待gen生成"

# 定义文件夹前缀和后缀
object_prefix = "Object"
object_suffix = "_Scan20W"
object_noisy_suffix = "_Scan20W_noisy"

# 假设文件夹从 Object1 到 ObjectN
num_objects = 111  # 例如，有10个对象：Object1, Object2, ..., Object10


for i in range(1, num_objects + 1):
    # 构建源文件夹路径
    object_dir = f"{object_prefix}{i}{object_suffix}"
    noisy_object_dir = f"{object_prefix}{i}{object_noisy_suffix}"

    # 构建源文件夹路径
    segm_folder = os.path.join(source_dir, object_dir, "segm")
    scan_settings_file = os.path.join(source_dir, object_dir, "scan settings.txt")
    log_folder = os.path.join(source_dir, noisy_object_dir, "log")
    volume_info_file = os.path.join(source_dir, noisy_object_dir, "volume_info.csv")

    # 检查源文件夹是否存在
    if not os.path.exists(segm_folder) or not os.path.exists(scan_settings_file) or not os.path.exists(log_folder) or not os.path.exists(volume_info_file):
        print(f"源文件夹或文件不存在，跳过 {object_dir} 和 {noisy_object_dir}。")
        continue  # 如果源文件夹或文件不存在，跳过当前循环

    # 目标文件夹路径
    target_folder = os.path.join(destination_dir, noisy_object_dir)

    # 检查目标文件夹是否已经存在，如果存在则跳过当前复制操作
    if os.path.exists(target_folder):
        print(f"文件夹 {target_folder} 已经存在, 跳过.")
        continue  # 跳过当前循环，进行下一个对象的处理

    # 如果目标文件夹不存在，则创建该文件夹
    os.makedirs(target_folder)
    print(f"------创建文件夹: {target_folder}")

    # 复制 segm 文件夹到目标文件夹
    if os.path.exists(segm_folder):
        target_segm_folder = os.path.join(target_folder, "segm")
        shutil.copytree(segm_folder, target_segm_folder)
        print(f"复制文件夹: {segm_folder} -> {target_segm_folder}")

    # 复制 scan settings.txt 文件到目标文件夹
    if os.path.exists(scan_settings_file):
        target_scan_settings_file = os.path.join(target_folder, "scan settings.txt")
        shutil.copy(scan_settings_file, target_scan_settings_file)
        print(f"复制文件: {scan_settings_file} -> {target_scan_settings_file}")

    # 复制 log 文件夹到目标文件夹
    if os.path.exists(log_folder):
        target_log_folder = os.path.join(target_folder, "log")
        shutil.copytree(log_folder, target_log_folder)
        print(f"复制文件夹: {log_folder} -> {target_log_folder}")

    # 复制 volume_info.csv 文件到目标文件夹
    if os.path.exists(volume_info_file):
        target_volume_info_file = os.path.join(target_folder, "volume_info.csv")
        shutil.copy(volume_info_file, target_volume_info_file)
        print(f"复制文件: {volume_info_file} -> {target_volume_info_file}")


print("所有文件已成功复制。")
