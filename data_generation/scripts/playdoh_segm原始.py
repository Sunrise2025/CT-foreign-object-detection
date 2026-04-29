import numpy as np
from pathlib import Path
import imageio
import shutil
from flexdata import data
from flextomo import projector
from flexcalc import process
from flexcalc import analyze
import cupy
import cupyx.scipy.ndimage
from skimage.util import img_as_ubyte
from natsort import natsorted
from termcolor import colored
from skimage.filters import threshold_multiotsu

import flexsim


def print_colored(text, color):
    print(colored(text, color))


def apply_median_filter(vol, size):
    """ Applies median filter to the volume using GPU acceleration (`cupyx.scipy.ndimage.median_filter`).
    """
    vol_gpu = cupy.asarray(vol)  # 将输入的体积数据转换为GPU数组
    vol_gpu = cupyx.scipy.ndimage.median_filter(vol_gpu, size)  # 使用GPU加速的中值滤波函数对体积数据进行处理
    vol_cpu = vol_gpu.get()  # 将处理后的体积数据从GPU拷贝回CPU

    mempool = cupy.get_default_memory_pool()  # 获取默认的GPU内存池
    pinned_mempool = cupy.get_default_pinned_memory_pool()
    mempool.free_all_blocks()  # 释放所有未使用的内存块，以防止内存泄漏
    pinned_mempool.free_all_blocks()

    return vol_cpu  # 返回处理后的体积数据


def reconstruct(input_folder, output_folder, bh_correction=True, compound='H2O',
                density=0.6):  # 使用 Flexbox 重建体积数据，支持强度校准，确保分割的精确度
    """Reconstructs the volume using flexbox. Supports beam-hardening correction, and that is crucial for good
    segmentation.
    """
    print('开始重建...')
    path = output_folder  # 将输出文件夹路径赋值给变量 path
    if bh_correction:  # 如果需要光束硬化校正
        save_path = path / "recon_bh"  # 设置保存路径为 "recon_bh
    else:
        save_path = path / "recon"  # 否则设置保存路径为 "recon"
    proj, geom = process.process_flex(input_folder, sample=2, skip=1, correct='cwi-flexray-2019-04-24')

    save_path.mkdir(exist_ok=True)

    geom.parameters['src_ort'] += (7 - 5.5)  # see below for flexbox hard coded values# 调整几何参数 src_ort 增加 1.5
    geom['det_roll'] -= 0.25  # 调整几何参数 det_roll 减少 0.25

    vol = projector.init_volume(proj)  # 初始化体积数据 vol，确保 vol 数组与投影数据的空间尺寸匹配
    # print(f'投影前playdoh_segm/proj_min: {proj.min()}, max: {proj.max()}')

    projector.FDK(proj, vol, geom)  # 使用 FDK 算法进行初步重建，并将结果存储在 vol 中
    print(f'投影后playdoh_segm/proj min: {proj.min()}, max: {proj.max()}')

    if bh_correction:  # 如果需要进行光束硬化校正
        density = density  # 将输入的 density 参数赋值给同名变量
        compound = compound  # 将输入的 compound 参数赋值给同名变量
        energy, spec = analyze.calibrate_spectrum(proj, vol, geom, compound=compound,
                                                  density=density)  # 移除源代码verbose=2，plot_path = save_path
        # 校准能谱并获取能量和谱线数据，使用 analyze.calibrate_spectrum 函数
        proj_cor = process.equivalent_density(proj, geom, energy, spec, compound=compound, density=density,
                                              preview=False)
        # 使用校准后的能谱和等效密度校正投影数据，使用 process.equivalent_density 函数
        vol_rec = np.zeros_like(vol)  # 创建一个与 vol 大小相同的空体积数组 vol_rec
        projector.FDK(proj_cor, vol_rec, geom)  # 使用校正后的投影数据进行重建，结果保存在 vol_rec 中
        vol = vol_rec  # 将 vol_rec 赋值给 vol，表示更新后的体积数据

    data.write_stack(save_path, 'slice', vol, dim=0)
    # 按第一个维度（通常是Z轴）切片提取，每隔1个切片保存一个。将重建的体积数据 vol 按照切片保存到指定路径 save_path这里是recon_bh
    print('重建结束...\n')


def segment(input_folder, output_folder, otsu_classes):  # 对重建的体积数据进行分割
    """Performs segmentation of the reconstructed volume.
    The number of classes for otsu should be e0000xplicitly given by the user.
    """
    print('开始分割...')
    path = output_folder  # 将输出文件夹路径赋值给变量 path
    recon_path = path / "recon_bh"  # 设置重建结果的路径为 "recon_bh"
    segm_path = path / "segm"  # 设置分割结果的路径为 "segm"
    segm_path.mkdir(exist_ok=True)  # 创建保存分割结果的文件夹，如果文件夹已经存在则不报错

    height = len(list(recon_path.glob("*.tiff")))  # 获取重建结果文件夹中所有 .tiff 文件的数量

    im_shape = (imageio.imread_v2(recon_path / "slice_{:06d}.tiff".format(0))).shape  # 读取第一张切片的形状，作为体积数据的形状

    vol = np.zeros((height, *im_shape))  # 创建一个空的体积数据数组 vol，数量为 height，切片形状为 *im_shape

    segm_vol = np.zeros_like(vol, dtype=np.uint8)  # 创建一个空的分割体积数据数组 segm_vol，与 vol 大小相同，数据类型为 uint8

    for i in range(height):
        vol[i, :] = imageio.imread_v2(recon_path / "slice_{:06d}.tiff".format(i))
        # 遍历所有切片文件，将它们读入 vol 数组中
    vol -= vol.min()    # 确保数据在 [0, 1] 范围, 将 vol 数组的最小值减去，以使最小值为 0
    vol /= vol.max()    # 将 vol 数组的最大值除以，以使最大值为 1
    vol = img_as_ubyte(vol)#归一化后的数据转换为 0 到 255 的整数范围
    vol = apply_median_filter(vol, 3)  # 对 vol 应用中值滤波，滤波器大小为 3
    thr = threshold_multiotsu(vol, classes=otsu_classes)
    #分割结果存储在一个 NumPy 数组中，其中每个元素的值代表该像素或体素的类别标签
    vol = np.array(vol)  # 确保 vol 是 NumPy 数组
    segm_vol = np.array(segm_vol)  # 确保 segm_vol 是 NumPy 数组
    thr = np.array(thr)  # 确保 thr 是 NumPy 数组
    # thr = skimage.filters.threshold_multiotsu(vol, classes=otsu_classes)
    # 使用多阈值 Otsu 分割法对 vol 进行分割，得到阈值数组 thr
    if otsu_classes == 3:
        print("警告Playdoh and Stone thresholds is: {:.2f} and {:.2f}".format(thr[0], thr[1]))  # 打印阈值img_as_ubyte 函数将数据缩放到 [0, 255] 的范围。这意味着当你将体积数据转换为 8 位无符号整型时，数据的原始浮点值（在 0 到 1 之间）会被线性地映射到整数值（0 到 255）
        segm_vol[vol > thr[0]] = 1  # 将 vol 中大于第一个阈值的像素设为 1
        segm_vol[vol > thr[1]] = 2  # 将 vol 中大于第二个阈值的像素设为 2
        print(f"Unique values in segmentation volume: {np.unique(segm_vol)}")

    if otsu_classes == 2:  # 如果 otsu_classes 为 2
        print("警告Playdoh threshold: {:.2f}".format(thr[0]))  # 打印阈值
        segm_vol[vol > thr[0]] = 1  # # 将 vol 中大于第一个阈值的像素设为 1

    for i in range(height):
        imageio.imwrite(segm_path / "slice_{:06d}.tiff".format(i), segm_vol[i, :])
        # 遍历所有切片，将分割结果保存到分割结果文件夹中，这是一个格式化的字符串，用于生成每个图像文件的名称。
        # {:06d} 是一个占位符，表示一个六位数的整数，不足六位的数字前面会被填充零。i 会被插入到这个占位符中，形成如
        # "slice_000001.tiff" 这样的文件名。
    print('分割结束...\n')


def preprocess_proj(input_folder, output_folder, skip_proj):  # 对投影应用暗场和亮场校正，并将其保存到单独的文件夹
    """Applies darkfield- and flatfield-correction to projections and saves them to a separate folder.
    """
    print('开始预处理...')
    path = input_folder  # 将输入文件夹路径赋值给变量 path
    out = output_folder  # 将输出文件夹路径赋值给变量 out
    log_path = out / "log"  # 设置保存路径为 "log"
    log_path.mkdir(exist_ok=True)  # 创建保存日志的文件夹，如果文件夹已经存在则不报错

    proj, flat, dark, geom = data.read_flexray(path, sample=2, skip=1, correct='cwi-flexray-2019-04-24')
    # 读取投影数据、亮场数据、暗场数据和几何信息
    proj = process.preprocess(proj, flat, dark)  # 对投影数据进行预处理，应用亮场和暗场校正
    proj = np.flip(proj, 0)  # 将投影数据沿第一个轴翻转

    for i in range(0, proj.shape[1], skip_proj):
        imageio.imwrite(log_path / "scan_{:06d}.tiff".format(i), proj[:, i, :])
        # 按照 skip_proj 的步长保存预处理后的投影数据
    print('预处理结束...\n')

#
# def check_intensity(output_folder):
#     """Computes mean value and standard deviations for materials based on the segmentation.
#     """  # 基于分割结果计算材料的平均值和标准差11
#     print('开始检测强度...')
#     path = Path(output_folder)  # 将输出文件夹路径赋值给变量 path
#     recon_path = path / "recon"  # 设置重建结果路径为 "recon"
#     segm_path = path / "segm"  # 设置分割结果路径为 "segm"
#
#     height = len(list(recon_path.glob("*.tiff")))  # 获取重建结果文件夹中所有 .tiff 文件的数量，即高度
#
#     im_shape = (imageio.v2.imread(recon_path / "slice_{:06d}.tiff".format(0))).shape
#     # 读取第一张切片的形状，作为体积数据的形状
#
#     vol = np.zeros((height, *im_shape), dtype=np.float32)
#     # 创建一个空的体积数据数组 vol，高度为 height，形状为 im_shape，数据类型为 float32
#
#     segm = np.zeros((height, *im_shape), dtype=np.int16)
#     # 创建一个空的分割体积数据数组 segm，高度为 height，形状为 im_shape，数据类型为 int16
#
#     for i in range(height):
#         vol[i, :] = imageio.v2.imread(recon_path / "slice_{:06d}.tiff".format(i))
#         segm[i, :] = imageio.v2.imread(segm_path / "slice_{:06d}.tiff".format(i))
#         # 遍历所有切片文件，将它们读入 vol 和 segm 数组中
#
#     playdoh_values = vol[segm == 1]
#     stone_values = vol[segm == 2]
#     playdoh_mean = vol[segm == 1].mean()
#     playdoh_std = vol[segm == 1].std()
#     stone_mean = vol[segm == 2].mean()
#     stone_std = vol[segm == 2].std()
#
#     if playdoh_values.size > 0:
#         playdoh_mean = playdoh_values.mean()
#         playdoh_std = playdoh_values.std()  # 计算 Playdoh  的平均强度和标准差
#     else:
#         print("Warning: No Playdoh values found!")
#         playdoh_mean = 0
#         playdoh_std = 0
#
#     if stone_values.size > 0:
#         stone_mean = stone_values.mean()
#         stone_std = stone_values.std()
#     else:
#         print("Warning: No Stone values found!")
#         stone_mean = 0
#         stone_std = 0
#
#     print("Playdoh mean intensity = {}".format(playdoh_mean))
#     print("Playdoh intensity std = {}".format(playdoh_std))
#     print("Stone mean intensity = {}".format(stone_mean))
#     print("Stone intensity std = {}".format(stone_std))
#     # 打印 Playdoh 和 Stone 的强度统计结果
#     print('强度检测结束...')


def multiple_objects_process():
    input_root = Path(config['Paths']['input_root'])
    output_root = Path(config['Paths']['output_root'])
    print(f"[输入数据根目录]  : {input_root.resolve()}")
    print(f"[输出数据根目录]  : {output_root.resolve()}")
    # 检查目录是否存在，不存在则创建（包括所有父目录）
    if not output_root.exists():
        output_root.mkdir(parents=True, exist_ok=True)
    sub_folders = []

    for path in input_root.iterdir():
        if path.is_dir():
            sub_folders.append(path.name)
    # 遍历输入根目录下的所有路径，如果是文件夹，则将文件夹名称添加到 sub_folders 列表中

    # 对 sub_folders 列表进行自然排序
    sub_folders = natsorted(sub_folders)

    # sub_folders = sorted(sub_folders)  # 对 sub_folders 列表进行排序
    input_folders = [input_root / sub_folder for sub_folder in sub_folders]
    output_folders = [output_root / sub_folder for sub_folder in sub_folders]
    #  output_folders = M:\m\pycharm_project\project1\Data\Real\Playdoh0-3\Playdoh0-3_segm\Object1_Scan20W
    assert len(input_folders) == len(output_folders)  # 确保输入文件夹和输出文件夹的数量相同

    # 初始化轮次计数器
    round_counter = 1
    # 遍历所有输入文件夹和对应的输出文件夹
    for i in range(len(input_folders)):
        # print(input_folders[i])  # 打印当前处理的输入文件夹路径
        if output_folders[i].exists():
            print(f"跳过{input_folders[i]} ，已经处理过，注意处理过的最后一个文件可能不完整.")
            continue
        text = f'第{round_counter}轮: 处理 {input_folders[i]}'
        print_colored(text, 'red')
        output_folders[i].mkdir(exist_ok=True)  # 创建对应的输出文件夹，如果文件夹已经存在则不报错
        # 遍历所有输入文件夹和对应的输出文件夹
        preprocess_proj(input_folders[i], output_folders[i], 5)  # 预处理投影数据，每隔5个投影保存一次,保存在log文件夹
        shutil.copy(input_folders[i] / 'scan settings.txt', output_folders[i])  # 复制扫描设置文件到输出文件夹，scan settings.txt
        reconstruct(input_folders[i], output_folders[i], bh_correction=True, compound='H2O', density=0.6)
        # 重建体积数据，进行光束硬化校正
        segment(input_folders[i], output_folders[i], 3)
        # 进行分割，使用3个 Otsu 类别        # Change the number of otsu classes depending on the presence of pebble stone in
        # the sample 根据样本中是否存在鹅卵石更改 Otsu 分割的类别数 segment(input_folders[i], output_folders[i], 2) 增加轮次计数器
        round_counter += 1


if __name__ == "__main__":  # 如果当前模块是直接运行的主程序（即通过命令行或者直接运行脚本），而不是被其他模块导入的，那么就执行 multiple_objects_process() 函数。
    config_fname = "playdoh.ini"  # 设置配置文件名
    config = flexsim.utils.read_config('playdoh.ini')  # 读取配置文件
    multiple_objects_process()
# CT 探测器的原始投影预处理（暗场校正 + 亮场校正，注意并未去噪），产生log文件夹；
# 将log文件夹中扫描的投影图片使用 FDK 算法进行重建为3D体积切片保存，产生recon_bh文件夹；
# 再将体积切片中值滤波（伪影（如束流硬化、噪声、环状结构））后进行分割标记类别，产生segm文件夹；
