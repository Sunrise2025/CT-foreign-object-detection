import os
import random
import shutil
import time

import imageio
import numpy as np
import tifffile
from matplotlib import pyplot as plt
from natsort import natsorted
from scipy import ndimage
from scipy.ndimage import binary_fill_holes
from skimage.util import img_as_ubyte
import cupy
import cupyx.scipy.ndimage
from pathlib import Path
from flexdata import data
from flextomo import projector
from flexcalc import process
from flexcalc import analyze
import flexsim
from skimage.filters import threshold_otsu, threshold_multiotsu
import numpy as np
import random


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
        fig.savefig(save_path)
        print(f"------Saved volume plot to: {save_path}")

def insert_slice(stone_1_max_shape, i, region, body_mask_3d, stone_1_data_3d, num_slices, min_position,insert_position):
    """
    height：切片的高度
    width：切片的宽度
    """
    insert_first_slices = insert_position
    insert_final_slices = insert_first_slices + num_slices
    print(f"【中心化】开始插入valid_region_3d切片次序从{insert_first_slices}到{insert_final_slices - 1}")

    mask = body_mask_3d[i]
    coords = np.argwhere(mask)
    min_row, min_col = coords.min(axis=0)
    max_row, max_col = coords.max(axis=0)
    print(f"【中心化】主体有效区域的范围: 行 {min_row}:{max_row}； 列 {min_col}:{max_col}")

    # 计算中心位置并固定
    center_row = round((min_row + max_row) / 2)  # round() 四舍五入得到最近的整数
    center_col = round((min_col + max_col) / 2)

    start_row = center_row - stone_1_max_shape[0] // 2
    start_col = center_col - stone_1_max_shape[1] // 2
    x1, y1 = min_position  # 异物原外接矩形左上角

    total_voxels = 0
    voxels_inside_body = 0
    voxels_outside_body = 0

    print(
        f"【中心化】主物体region 切片中心坐标=({center_row},{center_col}), "
        f"【中心化】所有异物所在的最大方形尺寸=({stone_1_max_shape[0]},{stone_1_max_shape[1]}), "
        f"【中心化】异物插入起点坐标=(start_row={start_row}, start_col={start_col})"
    )

    for k in range(insert_first_slices, insert_final_slices):  # k取值不包含insert_final_slices,即65-95
        region_slice = region[k]  # 当前切片
        stone_1_indices = np.argwhere(stone_1_data_3d[k - insert_first_slices] == 2)  # k - insert_first_slices范围0-30
        relative_indices = stone_1_indices - [x1, y1]  # 计算每个异物体素相对于 外接矩形左上角 (x1, y1) 的偏移

        for di, dj in zip(relative_indices[:, 0], relative_indices[:, 1]):
            rr = start_row + di
            cc = start_col + dj
            total_voxels += 1

            if ( 0 <= rr < region_slice.shape[0] and
                    0 <= cc < region_slice.shape[1]
                    #and body_slice[rr, cc]
                    ):
                region_slice[rr, cc] = 2
                if body_mask_3d[k, rr, cc]:
                    voxels_inside_body += 1
                else:
                    voxels_outside_body += 1
            else:
                raise ValueError(f"异物插入失败：目标坐标 ({rr},{cc}) 非法或不在有效区域")
        # print(f"valid_region_3d的第{k}层异物体素成功替换")
    print(f"异物整体插入后: 总体素={total_voxels}, 主体内={voxels_inside_body}, 凸出主体={voxels_outside_body}\n")
    return region


def playdoh_basic_augmentation(config_fname, input_folder, output_folder):
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

    vol0 = obj.get_volume().copy()
    print(f"[原始主体]vol0.shape = {vol0.shape}")

    labels0, nfeatures0 = ndimage.label(vol0 == 2)
    print(f"[原始主体] 主物体变换前体积体素总数: {np.sum(vol0 == 1)}")
    for i in range(1, nfeatures0 + 1):
        voxel_num = np.sum(labels0 == i)
        print(f"[原始异物] 原始异物 代号{i}：体素数 = {voxel_num}")
    print(f"[原始异物] 原始异物数量: {nfeatures0}")

    if nfeatures0 >= 2:
        voxel_counts = [(i + 1, np.sum(labels0 == (i + 1))) for i in range(nfeatures0)]
        voxel_counts.sort(key=lambda x: x[1], reverse=True)  # 按体素数量降序
        for lbl, _ in voxel_counts[1:]:
            vol0[labels0 == lbl] = 1  # 替换多余异物为主体
        labels0, nfeatures0 = ndimage.label(vol0 == 2)
    voxel_counts_after = [(i + 1, np.sum(labels0 == (i + 1))) for i in range(nfeatures0)]
    print(f"【保留1异物】 替换后异物数量: {nfeatures0}")
    for lbl, count in voxel_counts_after:
        print(f"【保留1异物】 保留异物 label={lbl}, 体素数量={count}")
    assert nfeatures0 == 1, "体积中必须已经有 1 个异物"
    print(f"【保留1异物】 主物体体素总数: {np.sum(vol0 == 1)}")

    print(f"【保留1异物】 vol0(after keep 1 FO).shape = {vol0.shape}\n")
    segm_folder = Path(output_folder) / "segm"
    os.makedirs(segm_folder, exist_ok=True)
    for i in range(vol0.shape[0]):
        slice_i = vol0[i].astype(np.uint8)
        slice_path = segm_folder / f"slice_{i:06d}.tiff"
        imageio.imwrite(slice_path, slice_i)
    obj.set_volume(vol0)

    folder_3d = Path(output_folder)
    plot_volume(vol0,
                title=output_folder.name,  # 子文件夹名作为标题
                save_path=folder_3d / f"{output_folder.name}.png")


def postprocess_proj( stone_1_path, output_folder):
    print(f"【输入路径】stone_1_path = {stone_1_path}")
    print(f"【输出路径】output_folder = {output_folder}")
    config_fname = "playdoh.ini"  # 设置配置文件名
    stone_1_path = Path(stone_1_path)
    os.makedirs(stone_1_path, exist_ok=True)
    playdoh_basic_augmentation(config_fname, stone_1_path, stone_1_path)
    # # 1. 读取Stone=0的Playdoh分割数据（假设每个文件夹包含多个tiff文件）
    # stone_1_dir = os.path.join(stone_1_path, 'segm')
    # stone_1_files = [f for f in os.listdir(stone_1_dir) if f.endswith('.tif') or f.endswith('.tiff')]
    # 2. 读取Stone=1的Playdoh数据（包含异物）
    stone_1_dir = os.path.join(stone_1_path, 'segm')
    stone_1_files = [f for f in os.listdir(stone_1_dir) if f.endswith('.tiff')]

    # 1. 获取所有异物切片中的异物最大尺寸
    stone_1_max_shape = (0, 0)
    min_position = (None, None)
    max_position = (None, None)
    min_row_curr, min_col_curr = float('inf'), float('inf')
    max_row_curr, max_col_curr = float('0'), float('0')

    for file in stone_1_files:
        stone_1_data0 = imageio.v2.imread(Path(stone_1_dir, file))
        if np.any(stone_1_data0 == 2):  # 只要 array 中有至少一个 True
            coordinates = np.argwhere(stone_1_data0 == 2)  # 如果这张切片中“至少存在一个异物像素（值为 2），那就把“所有异物像素的坐标”全部取出来。
            min_row, min_col = coordinates.min(axis=0)  # 对当前切片的每一列求最小值
            max_row, max_col = coordinates.max(axis=0)  # 对当前切片的每一列求最大值
            max_row_curr = max(max_row, max_row_curr)
            max_col_curr = max(max_col, max_col_curr)
            min_row_curr = min(min_row, min_row_curr)
            min_col_curr = min(min_col, min_col_curr)
            min_position = (min_row_curr, min_col_curr)  # 记录最大区域的左上角坐标
            max_position = (max_row_curr, max_col_curr)  # 记录最大区域的右下角坐标

            region_height = max_row - min_row + 1
            region_width = max_col - min_col + 1
            if region_height > stone_1_max_shape[0] or region_width > stone_1_max_shape[1]:
                stone_1_max_shape = (region_height, region_width)

    print(f"【异物定位】stone_1_data异物最大切片的尺寸: {stone_1_max_shape}")
    print(f"【异物定位】所有异物图像中的最大方形区域左上角坐标 min_position = {min_position}；右下角坐标: max_position = {max_position}")


    # 2. 提取Stone=1异物区域并补零得到新的stone_1_data
    stone_1_data_3d = []
    stone_1_data_files = []
    for file in stone_1_files:
        stone_1_data1 = imageio.v2.imread(Path(stone_1_dir, file))
        if np.any(stone_1_data1 == 2):
            # print(f"异物切片: {file}")
            new_data = np.zeros_like(stone_1_data1)  # 创建一个与原始图像大小相同的全零图像
            new_data[stone_1_data1 == 2] = 2  # 将原图中值为 2 的区域保留，stone_1_data == 2 会生成一个大小相同的布尔矩阵，每个元素是 True 或 False
            stone_1_data_3d.append(new_data)  # 将填充后的图像加入 3D 数据集
            stone_1_data_files.append(file)  # 存储有效文件名

    stone_1_data_3d = np.stack(stone_1_data_3d, axis=0)  # 将所有切片堆叠成一个 3D 数组
    # print("stone_1_data_3d 内容:", stone_1_data_3d)
    print("【异物定位】stone_1_data_3d 异物形状:", np.array(stone_1_data_3d).shape)
    num_slices = len(stone_1_data_3d)  # 异物切片的数量


    # 3. 对Stone=0的数据进行处理
    valid_region_3d = []
    invalid_region_3d = []
    valid_region_files = []  # 用于存储有效文件的文件名
    invalid_region_files = []

    for file0 in stone_1_files:
        stone_1_data = imageio.v2.imread(Path(stone_1_dir) / file0)
        if np.any(stone_1_data == 1):
            valid_region_3d.append(stone_1_data)
            valid_region_files.append(file0)  # 存储有效文件名
        else:
            invalid_region_3d.append(stone_1_data)
            invalid_region_files.append(file0)  # 存储无效文件名
    print(f"【主体定位】 stone_1_files count = {len(stone_1_files)}")
    print(f"【主体定位】 valid_region_3d slices = {len(valid_region_3d)}")
    print(f"【主体定位】 invalid_region_3d slices = {len(invalid_region_3d)}")
    print(f"【主体定位】 样本有效切片总数 = {len(valid_region_3d) + len(invalid_region_3d)}")

    valid_region_3d = np.array(valid_region_3d)  # 将valid_region_3d转为NumPy数组
    body_mask_3d = (valid_region_3d != 0)
    body_mask_3d = binary_fill_holes(body_mask_3d)

    insert_position = (len(valid_region_3d) - len(stone_1_data_3d)) // 2  # // 都是 直接向下取整，而不是四舍五入
    insert_position += 1  # 调整为中间部分后一个位置
    i = len(valid_region_3d) // 2
    print("【中心化前】异物体素数:", np.sum(stone_1_data_3d == 2))
    print(f"【中心化前】valid_region_3d 形状: {valid_region_3d.shape}，中间切片序号：{i}")
    # 在任何插入之前，先彻底清空旧异物
    valid_region_3d[valid_region_3d == 2] = 0
    stone_1_data_3d = insert_slice(stone_1_max_shape, i, valid_region_3d,  body_mask_3d, stone_1_data_3d, num_slices, min_position, insert_position)
    print("【中心化后】异物体素数:", np.sum(stone_1_data_3d == 2))
    print(f"【中心化后】非空气图像数据的形状: {stone_1_data_3d.shape}")

    # 中心化后的 3D 体
    centered_volume = stone_1_data_3d  # 实际就是 valid_region_3d
    plot_volume(
        centered_volume,
        title=f"{Path(output_folder).name}_centered",
        save_path=Path(output_folder) / f"{Path(output_folder).name}_centered.png"
    )

    # print(f"所有有效文件的文件名: {valid_region_files}") \
    # 5. 保存修改后的Playdoh数据（带异物），保持文件名一致
    output_dir = os.path.join(output_folder, 'segm')
    os.makedirs(output_dir, exist_ok=True)
    for idx, file0 in enumerate(valid_region_files):
        # 获取当前切片数据
        current_slice = stone_1_data_3d[idx]
        # print(f"[有效切片] 文件: {file0}, dtype: {current_slice.dtype}, unique values: {np.unique(current_slice)}")
        # 强制转换为 uint8
        current_slice = current_slice.astype(np.uint8)
        # 构造保存路径
        new_image_path = os.path.join(output_dir, file0)
        # 保存当前切片
        imageio.imwrite(new_image_path, current_slice)  # 保存每个切片为独立的图像
        # print(f"新处理切片已保存到：{new_image_path}")

    for idx, file0 in enumerate(invalid_region_files):
        current_slice = invalid_region_3d[idx]
        current_slice = current_slice.astype(np.uint8)
        new_image_path = os.path.join(output_dir, file0)
        imageio.imwrite(new_image_path, current_slice)  # 保存平滑后的数据
        # print(f"原切片已保存到：{new_image_path}")
    total_slices1 = len(valid_region_files)
    total_slices2 = len(invalid_region_files)
    print(f"新处理切片: {total_slices1} 张")
    print(f"原切片不需要处理: {total_slices2} 张")

    # 6. 复制扫描设置文件到输出文件夹
    shutil.copy(os.path.join(stone_1_path, 'scan settings.txt'), output_folder)

obj_id = 27
base_root = r"M:\m\pycharm_project\project1\Data\Real\Playdoh0-3"
stone_1_path = rf"{base_root}\Playdoh1\Object{obj_id}_Scan20W"
output_folder = rf"{base_root}\Playdoh基1类置中心\Object{obj_id}_Scan20W"
# 输出路径设置为stone_1_path目录
os.makedirs(output_folder, exist_ok=True)
start_time = time.time()
start_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time))
print(f"【开始时间】: {start_time_str}")
# 对每个输入文件夹进行后处理
postprocess_proj(stone_1_path, output_folder)
# 体积分割切片处理，生成虚拟异物样本
