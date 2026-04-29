import os
import pandas as pd
import re

# 定义源目录和目标目录
# source_dir = r"M:\m\pycharm_project\project1\Data\Real\Playdoh0-3\Playdoh0-3_segm"
source_dir = r"M:\m\pycharm_project\project1\Data\Real\Playdoh0-3\Training"
destination_dir =source_dir

# 定义文件夹前缀和后缀
object_prefix = "Object"
object_suffix = "_Scan20W"
# object_noisy_suffix = "_Scan20W_noisy"
# 假设文件夹从 Object1 到 ObjectN
num_objects = 300  # 例如，有10个对象：Object1, Object2, ..., Object10
# 定义一个空的 DataFrame 用于存储所有数据
all_data = pd.DataFrame()
# 遍历每个对象
for i in range(1, num_objects + 1):
    # 构建文件夹名称
    object_dir = f"{object_prefix}{i}{object_suffix}"# # 构建文件路径
    # noisy_object_dir = f"{object_prefix}{i}{object_noisy_suffix}"
    # volume_info_file = os.path.join(source_dir, noisy_object_dir, "volume_info.csv")
    volume_info_file = os.path.join(source_dir, object_dir, "volume_info.csv")

    # 检查 volume_info.csv 是否存在
    if not os.path.exists(volume_info_file):
        print(f"文件 {volume_info_file} 不存在，跳过 {object_dir}。")
        continue  # 如果文件不存在，跳过当前循环

    # 读取 volume_info.csv 文件，确保以逗号或制表符分隔
    try:
        volume_info_df = pd.read_csv(volume_info_file, sep=',', encoding='utf-8')  # 默认以逗号分隔
    except Exception as e:
        print(f"读取文件 {volume_info_file} 时出错: {e}")
        continue
    print(f"读取的列名: {volume_info_df.columns.tolist()}")
    # 去除列名中的多余空格或不可见字符
    volume_info_df.columns = volume_info_df.columns.str.strip()

    # 打印清理后的列名
    print(f"清理后的列名: {volume_info_df.columns.tolist()}")
    # 检查数据是否为空
    if volume_info_df.empty:
        print(f"{volume_info_file} 文件为空，跳过处理。")
        continue  # 跳过空文件

    # 将文件夹名称作为样品代号列添加到数据中
    volume_info_df['Sample_name'] = object_dir

    # 提取对象编号（ObjectX）中的数字部分
    volume_info_df['Object_number'] = volume_info_df['Sample_name'].apply(
        lambda x: int(re.search(r'Object(\d+)', x).group(1))  # 提取数字部分
    )

    # 将新的列 'Object_number' 移到最前面
    cols = ['Object_number'] + [col for col in volume_info_df.columns if col != 'Object_number']
    volume_info_df = volume_info_df[cols]

    # 只保留所需的列
    expected_columns = ['Object_number', 'Sample_name', 'Playdoh', 'Pebble', 'Pebble_objects', 'Sample_class']
    volume_info_df = volume_info_df[expected_columns]

    # 将当前对象的数据合并到总的 DataFrame 中
    all_data = pd.concat([all_data, volume_info_df], ignore_index=True)

    print(f"已处理：{object_dir}，当前合并数据：{volume_info_df.head()}")

# 检查合并后的数据
if all_data.empty:
    print("没有数据可保存，all_data 为空！")
else:
    # 获取目标目录的上一层目录
    # parent_dir = os.path.dirname(destination_dir)
    # 保存所有数据到一个总的 CSV 文件中，使用逗号分隔
    output_csv = os.path.join(destination_dir, "samples_info.csv")
    all_data.to_csv(output_csv, index=False, header=True, sep=',')  # 使用逗号分隔
    print(f"所有csv文件的数据已成功保存到 {output_csv}。")
