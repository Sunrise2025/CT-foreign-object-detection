import time

import numpy as np
from pathlib import Path
from scipy import ndimage
import skimage
from tqdm import tqdm
import flexsim
from scipy.ndimage import map_coordinates
from scipy.ndimage import gaussian_filter

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
def plot_volume(volume, title="Volume", save_path=None):
    """默认的 3D 视角（elev=30°, azim=-60°） —— 一种从左前上方观察的斜等测视角
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

    # plt.show()





def write_playdoh_volume_stats(obj, out_folder):  # 定义函数 write_playdoh_volume_stats，用于记录体积统计信息。
    mat_counts = obj.get_stats()  # mat_counts 保存 obj 对象中材料的统计信息。

    labels, nfeatures = ndimage.label(obj.volume == 2)
    # 使用 ndimage.label 对体积中值为2的部分进行连通区域标记，返回 labels 和 nfeatures（连通区域的数量）。labels是一个标记后的数组，其中每个连通区域（值为2的部分）都被分配了一个唯一的整数标签
    sample_class = nfeatures  # 将连通区域的数量赋值给 sample_class

    stat_line = "{},{},{}".format(",".join(str(num) for num in mat_counts), nfeatures, sample_class)
    # 格式化统计信息，生成 stat_line 字符串
    with open(out_folder / "volume_info.csv", "w") as f:
        # 打开 out_folder 中的 volume_info.csv 文件，准备写入，如果该文件不存在，Python 会自动创建这个文件
        f.write("Playdoh,Pebble,Pebble_objects,Sample_class\n")  # 写入CSV文件头部
        f.write(stat_line)  # 写入统计信息 stat_line
import numpy as np
from scipy.ndimage import label

def remove_stone_objects(obj, remove_fo):
    """
    Remove 'remove_fo' foreign objects from obj's volume,
    always removing the smallest ones first.
    All logic, logging, and操作都在这里，不调用外部函数。
    """
    # 获取当前体积
    volume = obj.get_volume()

    # 连通域标记
    labeled, nlabels = label(volume == 2)
    if nlabels == 0 or remove_fo <= 0:
        print("[异物移除算法] 没有异物或 remove_fo <= 0，跳过")
        return

    # 统计每个异物体素数
    voxel_counts = {i: int(np.sum(labeled == i)) for i in range(1, nlabels + 1)}
    print(f"[异物移除算法] 原始 (label, 体素数): {voxel_counts}")

    # 按体素数升序排序
    sorted_clusters = sorted(voxel_counts.items(), key=lambda x: x[1])
    print(f"[异物移除算法] 排序后 (label, 体素数): {sorted_clusters}")

    # 取最小的 remove_fo 个异物
    remove_clusters = sorted_clusters[:remove_fo]
    remove_labels = [lbl for lbl, _ in remove_clusters]
    print(f"[异物移除算法] 实际移除 label: {remove_labels}")

    # 执行删除
    for lb in remove_labels:
        volume[labeled == lb] = 1  # dest_num=1

    # 更新 obj 的体积
    obj.set_volume(volume)






def playdoh_basic_augmentation(config_fname, input_folder, out_subfolder, remove_fo=0):
    """ Create an artificial volume and the corresponding X-ray projections.不移除异物
    This function is used for replication and basic augmentation, it only allows FO modification.
    """
    config = flexsim.utils.read_config(config_fname)  # 读取配置文件 config_fname，内容存储在 config 中
    obj_vol_folder = input_folder / "segm"  # 设置对象体积文件夹路径 obj_vol_folder
    # out_folder = Path(config['Paths']['out_folder']) / out_subfolder
    # # 从playdoh.ini读取Paths中out_folder路径， 如果out_subfolder变量的值是subfolder_name，那么最终的输出文件夹路径将是C:\m\pycharm_project\project1\Data\Real\Playdoh2\Playdoh2_segm_Simulation\subfolder_name
    # out_folder.mkdir(parents=True, exist_ok=True)  # 确保自动创建父文件夹
    out_folder_str = config['Paths']['out_folder']
    # 将字符串转换为 Path 对象
    out_folder = Path(out_folder_str)
    out_folder = out_folder / f"Playdoh0-3_gen{num:03d}"
    num_angles = config['Simulation']['num_angles']  # 从配置文件中读取模拟的角度数量 num_angles
    energy_bins = config['Simulation']['energy_bins']  # 从配置文件中读取能量区间 energy_bins

    obj_shape = flexsim.utils.get_volume_properties(obj_vol_folder)  # 获取对象体积的属性 obj_shape
    mat = flexsim.MaterialHandler(energy_bins, config['Materials'])  # 创建材料处理器 mat，使用能量区间和配置文件中的材料信息
    obj = flexsim.ObjectCreator(obj_shape, mat)  # 创建对象生成器 obj，使用体积属性和材料处理器

    proj_shape = (obj_shape[0], num_angles, obj_shape[2])  # 设置投影形状 proj_shape
    noise = flexsim.NoiseModel(proj_shape, config['Noise'])  # 创建噪声模型 noise，使用投影形状和配置文件中的噪声信息
    proj = flexsim.Projector(obj, mat, noise, config['Simulation'])  # 创建投影器 proj，使用对象生成器、材料处理器和噪声模型
    proj.read_flexray_geometry(input_folder, (0, 360), 2)  # 读取 Flexray 几何配置

    obj.set_flexray_volume(obj_vol_folder)  # 设置 Flexray 体积
    if remove_fo != 0:  # 如果 remove_fo 不为0
        remove_stone_objects(obj, remove_fo)  # 移除指定数量的石头对象

    proj.create_projection(0, out_folder, 90)  # 创建 X 射线投影，并保存到输出文件夹
    obj.save_volume(out_folder)  # 保存对象体积到输出文件夹
    write_playdoh_volume_stats(obj, out_folder)  # 写入体积统计信息


def modify_main_object(obj):
    """ Object modification function that performs affine transformation of the whole object
    """
    scale = np.random.uniform(0.8, 1.2, size=(3,))  # 随机生成缩放参数
    shear = np.random.uniform(-0.2, 0.2, size=(3,))  # 随机生成剪切参数
    rotation = np.random.uniform(-90., 90., size=(3,))  # 随机生成旋转参数
    translation = (0., 0., 0.)  # 设置平移参数为 (0, 0, 0)
    print("Main object transformation: {}, {}, {}, {}".format(scale, shear, rotation, translation))  # 打印变换参数
    kwargs = {'mat_num': 3, 'scale': scale, 'shear': shear, 'rotation': rotation, 'translation': translation}  # 设置关键字参数
    obj.modify_volume(flexsim.transform.affine_volume, kwargs)  # 调用 modify_volume 方法，使用 affine_volume 进行仿射变换
    # 获取变换后的体积数据
    print(f"[DEBUG] Main object volume shape: {obj.get_volume().shape}, sum={np.sum(obj.get_volume()!=0)}")



def modify_foreign_object(volume,round_counter):
    """Object modification function that creates two FOs by performing affine transformations on one real FO.
    """
    # size = 6 because we generate parameters for two tranformations
    tmp_vol = np.zeros_like(volume, dtype=bool)
    tmp_vol[volume == 2] = True  # 原始异物
    res_vol = np.zeros_like(volume, dtype=int)
    res_vol[volume != 0] = 1  # 主体保留
    accepted_fo = 0  # ← 新增

    orig_voxels = np.sum(tmp_vol)
    scale = np.random.uniform(0.6, 1.8, size=(6,))  # 随机生成缩放参数，长度为6
    shear = np.random.uniform(-0.2, 0.2, size=(6,))  # 随机生成剪切参数，长度为6
    rotation = np.random.uniform(-10., 10., size=(6,))  # 随机生成旋转参数，长度为6
    translation1 = np.random.uniform(20., 60., size=(3,))  # 随机生成第一个平移参数
    translation2 = np.random.uniform(-60., -20., size=(3,))  # 随机生成第二个平移参数
    translation = np.concatenate((translation1, translation2))  # 将两个平移参数拼接在一起
    print("Foreign object transformation #1: {}, {}, {}, {}".format(scale[:3], shear[:3], rotation[:3],
                                                                    translation[:3]))  # 打印第一个变换参数
    print("Foreign object transformation #2: {}, {}, {}, {}".format(scale[3:], shear[3:], rotation[3:],
                                                              translation[3:]))  # 打印第二个变换参数
    for j in range(2):  # 遍历两个异物
        # 生成变换后的异物
        tmp_res = flexsim.transform.transform(
            tmp_vol.astype(float),
            scale=scale[3 * j:3 * (j + 1)],
            shear=shear[3 * j:3 * (j + 1)],
            rotation=rotation[3 * j:3 * (j + 1)],
            translation=translation[3 * j:3 * (j + 1)],
            interpolation='linear',
            rotation_units='deg',
            rotation_order='rzxz',
            device='cpu'
        )
        tmp_res = tmp_res.astype(bool)
        for k in range(2):
            tmp_res = skimage.morphology.binary_dilation(tmp_res)        # 二值膨胀

        tmp_res[volume == 0] = False        # 所有 volume 为 0 的位置，在 tmp_res 不能放，空气 → 不允许放异物

        # 存储已有异物的占用位置，用于碰撞检测
        occupied_mask = res_vol == 2
        if np.any(tmp_res & occupied_mask):
            print(f"[异物仿射变换] 异物 {j} 与已有异物重叠，重新生成")
            continue  # 重新生成本异物


        new_voxels = np.sum(tmp_res)
        if not (80  <= new_voxels <= 35500):
            print(f"[异物仿射变换] 异物 代号{j}体素异常: {new_voxels}")
            if j == 0:
                print("[异物仿射变换] 异物0异常，直接跳过本轮，不生成另一异物")
                return None  # 或者 break，看你函数逻辑
            continue
        else:
            print(f"[异物仿射变换] 异物 代号{j}体素: {new_voxels}")


        # 更新结果体积
        res_vol[tmp_res] = 2
        accepted_fo += 1
        # 强约束：输出连通域必须等于成功计数
    labels_out, nf_out = ndimage.label(res_vol == 2)
    print(f"[异物仿射变换] 当前 res_vol 中连通异物数量 nf_out = {nf_out}")
    for i in range(1, nf_out + 1):
        voxels = np.sum(labels_out == i)
        print(f"[异物仿射变换] 连通域 {i} 体素数 = {voxels}")

    if accepted_fo != 2 or nf_out != 2:
        return None
    return res_vol


def playdoh_triple_generation(config_fname,input_folder,num, output_subfolders):
    """ Create artificial volumes and the corresponding X-ray projections.
    This function is used for complex generation, it supports MO and FO modification
    """
    start_time = time.time()
    start_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time))
    print(f"【生成开始时间】: {start_time_str}/////////////Triple generation")  # 打印 "Triple generation"
    config = flexsim.utils.read_config(config_fname)  # 读取配置文件 config_fname，内容存储在 config 中
    obj_vol_folder = input_folder / "segm"
    # 设置对象体积文件夹路径 obj_vol_folder，input_folder是一个指向某个文件夹的Path对象，而"segm"是这个文件夹下的一个子文件夹名称
    out_folders = []  # 创建一个空列表，用于存储输出文件夹路径
    for subfolder in output_subfolders:  # 遍历输出子文件夹列表
        # subfolder= M:\m\pycharm_project\project1\Data\Real\Playdoh0-3\Playdoh0-3_segm_加噪音待gen生成\Object1_Scan20W\segm\slice_000000.tiff
        out_folder_str = config['Paths']['out_folder0']
        # 将字符串转换为 Path 对象
        out_folder = Path(out_folder_str)
        #out_folder  = M:\m\pycharm_project\project1\Data\Real\Playdoh0-3\Playdoh0-3_gen
        out_folder = out_folder / f"Playdoh0-3_gen{num:03d}_变1类置中心再生成"  / subfolder
        # 设置输出文件夹路径 out_folder，指向配置中指定的输出文件夹下的subfolder子文件夹
        out_folder.mkdir(parents=True, exist_ok=True)  # 确保自动创建父文件夹
        out_folders.append(out_folder)  # 将 out_folder 添加到 out_folders 列表中
    out_samples = len(out_folders)  # 获取输出样本的数量
    num_angles = config['Simulation']['num_angles']  # 从配置文件中读取模拟的角度数量 num_angles
    energy_bins = config['Simulation']['energy_bins']  # 从配置文件中读取能量区间 energy_bins
    obj_shape = flexsim.utils.get_volume_properties(obj_vol_folder)  # 获取分段切片对象体积的属性 obj_shape
    mat = flexsim.MaterialHandler(energy_bins, config['Materials'])  # 创建材料处理器 mat，使用能量区间和配置文件中的材料信息
    obj = flexsim.ObjectCreator(obj_shape, mat)  # 创建对象生成器 obj，使用体积属性和材料处理器
    proj_shape = (obj_shape[0], num_angles, obj_shape[2])  # 设置投影形状 proj_shape
    print("检查：config['Noise'] 配置部分:", config['Noise'])
    noise = flexsim.NoiseModel(proj_shape, config['Noise'])  # 创建噪声模型 noise，使用投影形状和配置文件中的噪声信息
    proj = flexsim.Projector(obj, mat, noise, config['Simulation'])  # 创建投影器 proj，使用对象生成器、材料处理器和噪声模型
    proj.read_flexray_geometry(input_folder, (0, 360), 2)  # 读取 Flexray 几何配置
    obj.set_flexray_volume(obj_vol_folder)  # 设置 Flexray 体积

    vol0 = obj.get_volume().copy()
    labels0, nfeatures0 = ndimage.label(vol0 == 2)
    for i in range(1, nfeatures0 + 1):
        voxel_num = np.sum(labels0 == i)
        print(f"[原始异物 ] 原始异物 代号{i}：体素数 = {voxel_num}")
    print(f"[原始异物 ] 原始异物数量: {nfeatures0}")

    if nfeatures0 >= 2:
        voxel_counts = [(i + 1, np.sum(labels0 == (i + 1))) for i in range(nfeatures0)]
        voxel_counts.sort(key=lambda x: x[1], reverse=True)  # 按体素数量降序
        for lbl, _ in voxel_counts[1:]:
            vol0[labels0 == lbl] = 1  # 替换多余异物为主体
        labels0, nfeatures0 = ndimage.label(vol0 == 2)
    voxel_counts_after = [(i + 1, np.sum(labels0 == (i + 1))) for i in range(nfeatures0)]
    print(f"[保留异物] 替换后异物数量: {nfeatures0}")
    for lbl, count in voxel_counts_after:
        print(f"[保留异物] 保留异物 label={lbl}, 体素数量={count}")
    assert nfeatures0 == 1, "体积中必须已经有 1 个异物"
    print(f"[原始主体] 主物体变换前体积形状: {vol0.shape}")
    print(f"[原始主体] 主物体变换前体积体素总数: {np.sum(vol0 == 1)}")
    print(" —— —— 主物体仿射变换前，各异物体素数——  ——")

    obj.set_volume(vol0)

    modify_main_object(obj)  # 调用 modify_main_object 方法，进行主对象变换
    # 双异物仿射 → 更新 obj.volume

    vol1 = obj.get_volume()
    labels1, nfeatures1 = ndimage.label(vol1 == 2)
    print(f"[主物体仿射变换] 主物体变换后体积形状: {vol1.shape}")
    print(f"[主物体仿射变换] 主物体变换后体积体素总数: {np.sum(vol1 == 1)}")
    print(f"[主物体仿射变换] 主物体变换后检测到的异物数量: {nfeatures1}")
    print(" —— ——主物体仿射变换后，各异物体素数—— ——")

    for i in range(1, nfeatures1 + 1):
        voxel_num = np.sum(labels1 == i)
        print(f"[主物体仿射变换] 变换后异物 {i}：体素数 = {voxel_num}")

    volume = obj.get_volume()
    volume_save = volume.copy()  # 保存初始体积
    round_counter = 1
    failed_attempts = 0  # 新增失败计数
    max_failed_attempts = 15  # 容忍阈值

    while True:
        print(" —— ——异物仿射变换前—— ——")
        new_volume = modify_foreign_object(volume_save,round_counter)  # 内部会生成异物
        if new_volume is None:
            round_counter += 1
            failed_attempts += 1
            print(f"[异物仿射变换] 异物生成失败，连续失败次数 {failed_attempts}")
            if failed_attempts >= max_failed_attempts:
                print(f"++++++++[异物仿射变换] 连续 {failed_attempts} 次生成失败，重新变换主对象体积")
                modify_main_object(obj)  # 重新变换主对象

                vol1 = obj.get_volume()
                labels1, nfeatures1 = ndimage.label(vol1 == 2)
                print(f"[主物体仿射变换-连续 {failed_attempts} 次生成失败] 主物体变换后体积形状: {vol1.shape}")
                print(f"[主物体仿射变换-连续 {failed_attempts} 次生成失败] 主物体变换后体积体素总数: {np.sum(vol1 == 1)}")
                print(f"[主物体仿射变换-连续 {failed_attempts} 次生成失败] 主物体变换后检测到的异物数量: {nfeatures1}")
                print(" —— ——主物体仿射变换后，各异物体素数—— ——")
                for i in range(1, nfeatures1 + 1):
                    voxel_num = np.sum(labels1 == i)
                    print(f"[主物体仿射变换] 变换后异物 {i}：体素数 = {voxel_num}")

                failed_attempts = 0  # 重置失败计数
            continue
        labels, nfeatures = ndimage.label(new_volume == 2)
        print(f"[异物仿射变换] 异物变换后检测到 {nfeatures} 个异物")
        for i in range(1, nfeatures + 1):
            voxel_num = np.sum(labels == i)
            print(f"[异物仿射变换] 新生成异物 代号{i} 体素数 = {voxel_num}")
        labels, nfeatures = ndimage.label(new_volume == 2)
        if nfeatures == 2:
            print(f"***********成功生成含有2个异物的体积")

        sample_basename = output_subfolders[0].rsplit('_', 1)[0]
        folder_str = Path(config['Paths']['out_folder0'])/ f"Playdoh0-3_gen{num:03d}_变1类置中心再生成"
        folder_3d = folder_str / "3D图"
        if not folder_3d.exists():
            folder_3d.mkdir(exist_ok=True)  # 不创建父目录
        plot_volume(new_volume,
                    title=sample_basename,  # 子文件夹名作为标题
                    save_path=folder_3d / f"{sample_basename}.png")

        # plot_volume(new_volume, title=f"After FO transformation")
        break

    obj.set_volume(new_volume)
    volume_save = obj.get_volume()  # 保存当前体积
    for i in range(out_samples):  # 遍历输出样本数量
        obj.set_volume(volume_save)  # 恢复保存的体积
        if i != 0:  # 如果不是第一个样本

            remove_stone_objects(obj, i)  # 调用 remove_stone_objects 方法，移除指定数量的石头对象

            volume_after = obj.get_volume()
            labels_after, nfeatures_after = ndimage.label(volume_after == 2)
            voxel_counts_after = [(lbl, np.sum(labels_after == lbl)) for lbl in range(1, nfeatures_after + 1)]
            print(f"[移除算法] 第{i}轮，剩余异物: {voxel_counts_after}\n")


        proj.create_projection(0, out_folders[i], 90)
        # 创建 scan扫描的x射线投影图片,对投影 & 亮场添加混合泊松-高斯噪声，保存在log文件夹，并保存到输出文件夹
        obj.save_volume(out_folders[i])  # 保存保存对象体积"Volume"文件到输出文件夹
        write_playdoh_volume_stats(obj, out_folders[i])  # 写入体积统计信息


def batch_generation(input_root, config_fname,num, generation_base_samples):
    """Create new X-ray images by modifying the main and foreign object.
    The number of artificial volumes to generate is taken from the config file.
    """
    # 批量生成函数，通过修改主对象和外来对象创建新的 X 射线图像
    # 生成的人工体积数量取自配置文件
    config = flexsim.utils.read_config(config_fname)  # 读取配置文件 config_fname，内容存储在 config 中
    # The samples will be generated in groups of 3, so //3
    augmentation_samples = config['Simulation']['augmentation_samples'] // 3
    # 样本将以 3 为一组生成，这里最终值90/3=30
    for sample in generation_base_samples:  # 遍历基础样本列表
        path = input_root / sample  # 构造一个指向Object1_Scan20W 的路径
        data = np.loadtxt(path / 'volume_info.csv', skiprows=1, delimiter=',', dtype=int)  # 读取 volume_info.csv 文件数据
        sample_class = data[-1]  # 获取样本类别
        assert sample_class >= 1 # 断言样本类别为1

    for sample in generation_base_samples:  # 遍历基础样本列表
        sample_basename = sample.split('_')[0]  # 获取样本基本名称，如Object57_Scan20W_noisy显示Object57_Scan20W
        print(f"---------------基本外形样本对象:{sample_basename}-----------")  # 打印样本基本名称如Object57_Scan20W
        for i in tqdm(range(augmentation_samples)):  # 遍历增强样本数量，显示进度条
            aug_name = "{}_mod{:03d}".format(sample_basename, i)  # 格式化增强样本名称，如Object57_Scan20W_mod001
            out_folders = ["{}_2fo".format(aug_name), "{}_1fo".format(aug_name), "{}_0fo".format(aug_name)]
            # 设置输出文件夹名称,如Object57_Scan20W_mod001_2fo1
            playdoh_triple_generation(config_fname, input_root / sample, num, out_folders)  # 调用 playdoh_triple_generation 方法

# M:\m\pycharm_project\project1\Data\Real\Playdoh0-3\Playdoh0-3_segm_加噪音待gen生成\Object1_Scan20W
  # generation_base_samples = ['Object2_Scan20W_noisy'] ，   playdoh_triple_generation(config_fname, input_root / sample, out_folders)


if __name__ == "__main__":

    config_fname = "playdoh.ini"  # 设置配置文件名
    config = flexsim.utils.read_config('playdoh.ini')  # 读取配置文件
    input_root = Path(config['gen_Paths']['input_root'])
    out_root = Path(config['Paths']['out_folder0'])
    print(f"[输入数据根目录]  : {input_root.resolve()}")
    print(f"[输出数据根目录]  : {out_root.resolve()}")
    np.random.seed(seed=6)  # 设置随机数种子
    num = config['General']['samples_num']
    num = int(num)  # 将 patience 转换为整数类型
    base_sample_name = 'Object{}_Scan20W'
    # 假设你希望用 samples_num 来替换 'Object13_Scan20W_noisy' 中的 '13'
    generation_base_samples = [base_sample_name.format(num)]
    # 从配置文件中读取 out_folder 路径字符串
    # base_sample_name = 'Object{}_Scan20W_noisy'
    # generation_base_samples = ['Object2_Scan20W_noisy']  # 设置基础样本列表,类别1含1个异物
    batch_generation(input_root, config_fname, num,generation_base_samples)